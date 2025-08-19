import torch
import torch.nn as nn
from torch_geometric.datasets import MoleculeNet # Corrected import
import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
import logging
from sklearn.metrics import roc_auc_score

from neurospiral.parse_args import args
from neurospiral.logger import setup_logger
from neurospiral.molecule_process import ATOM_FEATURE_DIM, NUM_SUBSTRUCTURE_TYPES, BOND_FEATURE_DIM, GLOBAL_FEATURE_DIM
from neurospiral.model import MultiLevelGNN
from neurospiral.prepare_data import prepare_train_val_test_set

def train(loader, epoch_num=0):
    model.train()
    total_loss = 0
    for data in tqdm(loader, desc=f"Training Epoch {epoch_num}"):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        
        mask = ~torch.isnan(data.y)
        
        if mask.sum() > 0:
            loss = criterion(out[mask], data.y[mask])
            total_loss += loss.mean().item() * data.num_graphs
            loss.mean().backward()
            optimizer.step()
    return total_loss / len(loader.dataset)

def evaluate(loader):
    model.eval()
    y_preds_all = []
    y_trues_all = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            y_pred = torch.sigmoid(out)
            
            for i in range(data.y.shape[0]):
                for j in range(data.y.shape[1]):
                    if not torch.isnan(data.y[i, j]):
                        y_preds_all.append(y_pred[i, j].item())
                        y_trues_all.append(data.y[i, j].item())
    
    if len(y_trues_all) == 0 or len(set(y_trues_all)) < 2:
        return 0.5
    
    try:
        auc = roc_auc_score(y_trues_all, y_preds_all)
    except ValueError:
        auc = 0.5
    return auc

def evaluate_per_task(loader):
    model.eval()

    all_y_trues_per_task = [[] for _ in range(NUM_TASKS)]
    all_y_preds_per_task = [[] for _ in range(NUM_TASKS)]

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            y_pred_batch = torch.sigmoid(out)

            for i in range(data.y.shape[0]):
                for j in range(NUM_TASKS):
                    true_label = data.y[i, j]
                    predicted_prob = y_pred_batch[i, j]

                    if not torch.isnan(true_label):
                        all_y_trues_per_task[j].append(true_label.item())
                        all_y_preds_per_task[j].append(predicted_prob.item())
    
    task_aucs = {}
    for task_idx in range(NUM_TASKS):
        y_trues = all_y_trues_per_task[task_idx]
        y_preds = all_y_preds_per_task[task_idx]

        current_task_name = f"Task_{task_idx}"

        if len(y_trues) < 2 or len(set(y_trues)) < 2:
            task_aucs[current_task_name] = 0.5 
        else:
            try:
                auc = roc_auc_score(y_trues, y_preds)
                task_aucs[current_task_name] = auc
            except ValueError:
                task_aucs[current_task_name] = 0.5
    
    return task_aucs

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    print(f"Arguments: {args}")
    set_seed(args.seed)
    args.dataset = 'tox21'
    if not os.path.exists(args.result_folder+ '/'+args.dataset):
        os.makedirs(args.result_folder+ '/'+args.dataset)
    setup_logger(args.result_folder+'/'+args.dataset+datetime.now().strftime('/log_%Y%m%d_%H%M%S.txt'), silent=False)

    print("Loading dataset...")
    dataset = MoleculeNet(root='./data', name='Tox21')
    print(f"Dataset loaded. Number of molecules: {len(dataset)}")
    print(f"Number of tasks: {dataset.num_classes}")
    NUM_TASKS = dataset.num_classes
    filtered_dataset = []
    for data_item in dataset:
        if data_item.y is not None and not torch.isnan(data_item.y).all():
            filtered_dataset.append(data_item)
    print(f"Filtered dataset (valid labels): {len(filtered_dataset)} molecules.")

    train_loader, val_loader, test_loader = prepare_train_val_test_set(filtered_dataset, batch_size=32)

    SUB_FEATURE_DIM_ORIGINAL = ATOM_FEATURE_DIM + NUM_SUBSTRUCTURE_TYPES
    HIDDEN_CHANNELS = 64
    OUT_CHANNELS = 64 
    GNN_HEADS = 4

    assert HIDDEN_CHANNELS % GNN_HEADS == 0, "HIDDEN_CHANNELS must be divisible by GNN_HEADS"
    assert OUT_CHANNELS % GNN_HEADS == 0, "OUT_CHANNELS must be divisible by GNN_HEADS"

    model = MultiLevelGNN(ATOM_FEATURE_DIM, SUB_FEATURE_DIM_ORIGINAL, HIDDEN_CHANNELS, OUT_CHANNELS, NUM_TASKS, BOND_FEATURE_DIM, GLOBAL_FEATURE_DIM, gnn_heads=GNN_HEADS)
    if args.gpu>=0:
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu') # 'cuda' if torch.cuda.is_available() else 'cpu')

    if not args.eval:
        model.to(device)
    else:
        try:
            # 确保加载到正确的设备
            model.load_state_dict(torch.load(args.best_model_path, map_location=device))
            model.to(device) # 将模型放到指定设备
            print(f"Model successfully loaded from {args.best_model_path} and set to evaluation mode.")
        except FileNotFoundError:
            print(f"Error: Model file not found at {args.best_model_path}. Please ensure the model was saved correctly.")
            exit()
        except Exception as e:
            print(f"Error loading model: {e}")
            exit()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    if not args.eval:
        epochs = args.n_epochs
        print(f"\nStarting training for {epochs} epochs on {device}...")
        for epoch in range(1, epochs + 1):
            loss = train(train_loader, epoch_num=epoch)
            train_auc = evaluate(train_loader)
            val_auc = evaluate(val_loader)
            logging.info(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}')
            if epoch % args.save_epoch_int == 0:
                if not os.path.exists(args.model_folder+ '/'+args.dataset):
                    os.makedirs(args.model_folder+ '/'+args.dataset)
                model_save_path = args.model_folder + '/'+args.dataset+f'/model_epoch_{epoch}' + '.pth'
                torch.save(model.state_dict(), model_save_path)
                logging.info(f"Model saved to {model_save_path}")
    else:
        test_auc = evaluate(test_loader)
        print(f'\nTest AUC: {test_auc:.4f}')