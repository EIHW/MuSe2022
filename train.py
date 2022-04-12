import time
import os
import numpy as np
import torch
import torch.optim as optim

from eval import evaluate


def train(model, train_loader, optimizer, loss_fn, use_gpu=False):

    report_loss, report_size = 0, 0
    total_loss, total_size = 0, 0

    model.train()
    if use_gpu:
        model.cuda()

    for batch, batch_data in enumerate(train_loader, 1):
        features, feature_lens, labels, metas = batch_data
        batch_size = features.size(0)

        if use_gpu:
            features = features.cuda()
            feature_lens = feature_lens.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()

        preds = model(features, feature_lens)

        loss = loss_fn(preds.squeeze(-1), labels.squeeze(-1), feature_lens)

        loss.backward()
        optimizer.step()

        report_loss += loss.item() * batch_size
        report_size += batch_size

        total_loss += report_loss
        total_size += report_size
        report_loss, report_size, start_time = 0, 0, time.time()

    train_loss = total_loss / total_size
    return train_loss


def save_model(model, model_folder, current_seed):
    model_file_name = f'model_{current_seed}.pth'
    model_file = os.path.join(model_folder, model_file_name)
    torch.save(model, model_file)
    return model_file


def train_model(task, model, data_loader, epochs, lr, model_path, current_seed, use_gpu, loss_fn, eval_fn,
                eval_metric_str, early_stopping_patience, reduce_lr_patience, regularization=0.0):
    train_loader, val_loader, test_loader = data_loader['train'], data_loader['devel'], data_loader['test']

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=regularization)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=reduce_lr_patience,
                                                        factor=0.5, min_lr=1e-5, verbose=True)
    best_val_loss = float('inf')
    best_val_score = -1
    best_model_file = ''
    early_stop = 0

    for epoch in range(1, epochs + 1):
        print(f'Training for Epoch {epoch}...')
        train_loss = train(model, train_loader, optimizer, loss_fn, use_gpu)
        val_loss, val_score = evaluate(task, model, val_loader, loss_fn=loss_fn, eval_fn=eval_fn, use_gpu=use_gpu)

        print(f'Epoch:{epoch:>3} / {epochs} | [Train] | Loss: {train_loss:>.4f}')
        print(f'Epoch:{epoch:>3} / {epochs} | [Val] | Loss: {val_loss:>.4f} | [{eval_metric_str}]: {val_score:>7.4f}')
        print('-' * 50)

        if val_score > best_val_score:
            early_stop = 0
            best_val_score = val_score
            best_val_loss = val_loss
            best_model_file = save_model(model, model_path, current_seed)

        else:
            early_stop += 1
            if early_stop >= early_stopping_patience:
                print(f'Note: target can not be optimized for {early_stopping_patience} consecutive epochs, '
                      f'early stop the training process!')
                print('-' * 50)
                break

        lr_scheduler.step(1 - np.mean(val_score))

    print(f'Seed {current_seed} | '
          f'Best [Val {eval_metric_str}]:{best_val_score:>7.4f} | Loss: {best_val_loss:>.4f}')
    return best_val_loss, best_val_score, best_model_file
