import argparse
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from dataloader import *
from model import *



def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--learning_rate', type=float, default=5e-4)
    parser.add_argument('-e', '--epoch', type=int, default=129)
    parser.add_argument('-n', '--model_name', type=str, default='split10_triplet_fast')
    parser.add_argument('-t', '--training_data', type=str, default='split10')
    parser.add_argument('-d', '--hidden_dim', type=int, default=256)
    parser.add_argument('-o', '--out_dim', type=int, default=128)
    parser.add_argument('--adaptive_rate', type=int, default=10)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--train_txt', type=str, default='train.txt',
                        help='训练集txt路径，对应h5默认是同目录同名 .h5')
    parser.add_argument('--valid_txt', type=str, default='valid.txt',
                        help='验证集txt路径，对应h5默认是同目录同名 .h5')
    parser.add_argument('--pool_txt', type=str, default='a.txt',
                        help='pool集txt路径，对应h5默认是同目录同名 .h5')
    parser.add_argument('--theta', type=int, default=40,
                        help='序列截断/补零长度')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--knn', type=int, default=10)
    parser.add_argument('--mine_chunk_size', type=int, default=1024,
                        help='分块挖硬负样本时的anchor块大小')
    args = parser.parse_args()
    return args



def duibiloss(anchor, positive, negative, margin=1.0):
    pos_dist = torch.norm(anchor - positive, dim=(1, 2))
    neg_dist = torch.norm(anchor - negative, dim=(1, 2))
    loss = torch.relu(pos_dist - neg_dist + margin)
    return loss.mean()



def plot_losses(train_losses, val_losses, save_path='./result/loss_plot.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='训练损失')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='验证损失')
    plt.xlabel('训练轮次')
    plt.ylabel('损失值')
    plt.title('训练和验证损失变化曲线')
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f'损失曲线图已保存到: {save_path}')



def save_loss_data(train_losses, val_losses, args, save_dir='./result'):
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f'{args.model_name}_loss_data_{timestamp}.pkl'
    save_path = os.path.join(save_dir, filename)
    loss_data = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'args': vars(args),
        'timestamp': timestamp,
    }
    os.makedirs(save_dir, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(loss_data, f)
    print(f'损失数据已保存到: {save_path}')



def stack_feature_tensor(names, feature_dict):
    return torch.stack([
        torch.from_numpy(feature_dict[name]).float() if isinstance(feature_dict[name], np.ndarray)
        else feature_dict[name].float()
        for name in names
    ], dim=0)



def build_loader(dataset, batch_size, shuffle, num_workers, pin_memory=True):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )



def embed_tensor_in_batches(model, tensor_cpu, device, batch_size=2048):
    model.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, tensor_cpu.shape[0], batch_size):
            batch = tensor_cpu[start:start + batch_size].to(device, non_blocking=True)
            outs.append(model(batch))
    return torch.cat(outs, dim=0)



def mine_hard_negative_cross_fast(
    emb_anchor,
    emb_pool,
    anchor_ids,
    pool_ids,
    anchor_labels,
    pool_labels,
    knn=10,
    chunk_size=1024,
):
    """
    GPU/向量化版硬负样本挖掘。
    对几千规模数据，通常会比原来的双层 Python 循环快很多。
    """
    device = emb_anchor.device
    anchor_flat = emb_anchor.reshape(emb_anchor.shape[0], -1).contiguous()
    pool_flat = emb_pool.reshape(emb_pool.shape[0], -1).contiguous()

    anchor_labels_t = torch.as_tensor(anchor_labels, device=device, dtype=torch.long)
    pool_labels_t = torch.as_tensor(pool_labels, device=device, dtype=torch.long)

    k = min(knn, pool_flat.shape[0])
    neg_dict = {}
    eps = 1e-6

    for start in range(0, anchor_flat.shape[0], chunk_size):
        end = min(start + chunk_size, anchor_flat.shape[0])
        a_block = anchor_flat[start:end]
        a_labels = anchor_labels_t[start:end]

        dists = torch.cdist(a_block, pool_flat, p=2)
        same_label_mask = a_labels[:, None].eq(pool_labels_t[None, :])
        dists = dists.masked_fill(same_label_mask, float('inf'))

        topk_dist, topk_idx = torch.topk(dists, k=k, dim=1, largest=False)
        topk_dist = topk_dist.detach().cpu()
        topk_idx = topk_idx.detach().cpu()

        for row_idx, aid in enumerate(anchor_ids[start:end]):
            row_dist = topk_dist[row_idx].tolist()
            row_idxs = topk_idx[row_idx].tolist()

            valid_pairs = [
                (pool_ids[j], float(d))
                for j, d in zip(row_idxs, row_dist)
                if np.isfinite(d)
            ]

            if not valid_pairs:
                neg_dict[aid] = {'negative': [], 'weights': []}
                continue

            neg_ids = [pid for pid, _ in valid_pairs]
            dist_vals = np.asarray([d for _, d in valid_pairs], dtype=np.float64)
            inv = 1.0 / (dist_vals + eps)
            weights = (inv / inv.sum()).astype(np.float64).tolist()
            neg_dict[aid] = {
                'negative': neg_ids,
                'weights': weights,
            }

    return neg_dict



def rebuild_negatives(model, anchor_tensor_cpu, pool_tensor_cpu,
                      anchor_ids, pool_ids, anchor_labels, pool_labels,
                      device, knn, chunk_size):
    with torch.no_grad():
        emb_anchor = embed_tensor_in_batches(model, anchor_tensor_cpu, device)
        emb_pool = embed_tensor_in_batches(model, pool_tensor_cpu, device)
        return mine_hard_negative_cross_fast(
            emb_anchor=emb_anchor,
            emb_pool=emb_pool,
            anchor_ids=anchor_ids,
            pool_ids=pool_ids,
            anchor_labels=anchor_labels,
            pool_labels=pool_labels,
            knn=knn,
            chunk_size=chunk_size,
        )



def run_epoch(model, loader, optimizer, device, dtype, loss_fn, train_mode=True, verbose=False, epoch=0):
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    n_batches = 0
    start_time = time.time()

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for batch_idx, (anchor, positive, negative) in enumerate(loader):
            anchor = anchor.to(device=device, dtype=dtype, non_blocking=True)
            positive = positive.to(device=device, dtype=dtype, non_blocking=True)
            negative = negative.to(device=device, dtype=dtype, non_blocking=True)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)

            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)
            loss = loss_fn(anchor_out, positive_out, negative_out)

            if train_mode:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if verbose and batch_idx % 20 == 0:
                ms_per_batch = (time.time() - start_time) * 1000 / max(1, (batch_idx + 1))
                print(
                    f'| epoch {epoch:03d} | batch {batch_idx:05d}/{len(loader):05d} '
                    f'| ms/batch {ms_per_batch:8.2f} | loss {loss.item():.4f}'
                )

    return total_loss / max(1, n_batches)



def main():
    seed_everything()
    ensure_dirs('./result')
    args = parse()

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    dtype = torch.float32
    torch.backends.cudnn.benchmark = torch.cuda.is_available()

    print('==> 使用设备:', device, '| 数据类型:', dtype)
    print('==> 参数:', args)

    pre_feas_train, labels_train, names_train, _ = data_pre(args.train_txt, theta=args.theta)
    pre_feas_val, labels_val, names_val, _ = data_pre(args.valid_txt, theta=args.theta)
    pre_feas_pool, labels_pool, names_pool, _ = data_pre(args.pool_txt, theta=args.theta)

    id_label_train, label_id_train = get_label_id_dict(names_train, labels_train)
    label_id_train = {key: list(label_id_train[key]) for key in label_id_train.keys()}
    id_label_val, label_id_val = get_label_id_dict(names_val, labels_val)
    label_id_val = {key: list(label_id_val[key]) for key in label_id_val.keys()}
    id_label_pool, label_id_pool = get_label_id_dict(names_pool, labels_pool)
    label_id_pool = {key: list(label_id_pool[key]) for key in label_id_pool.keys()}

    train_ids = list(names_train)
    val_ids = list(names_val)
    pool_ids = list(names_pool)

    pre_feas_tensor_train = stack_feature_tensor(train_ids, pre_feas_train)
    pre_feas_tensor_val = stack_feature_tensor(val_ids, pre_feas_val)
    pre_feas_tensor_pool = stack_feature_tensor(pool_ids, pre_feas_pool)

    model = LayerNormNet(args.hidden_dim, args.out_dim, device, dtype).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    loss_fn = duibiloss

    print('==> 初始硬负样本挖掘(train)')
    neg_dict_train = rebuild_negatives(
        model=model,
        anchor_tensor_cpu=pre_feas_tensor_train,
        pool_tensor_cpu=pre_feas_tensor_pool,
        anchor_ids=train_ids,
        pool_ids=pool_ids,
        anchor_labels=labels_train,
        pool_labels=labels_pool,
        device=device,
        knn=args.knn,
        chunk_size=args.mine_chunk_size,
    )

    print('==> 初始硬负样本挖掘(valid)')
    neg_dict_val = rebuild_negatives(
        model=model,
        anchor_tensor_cpu=pre_feas_tensor_val,
        pool_tensor_cpu=pre_feas_tensor_pool,
        anchor_ids=val_ids,
        pool_ids=pool_ids,
        anchor_labels=labels_val,
        pool_labels=labels_pool,
        device=device,
        knn=args.knn,
        chunk_size=args.mine_chunk_size,
    )

    train_dataset = TripletDatasetCross(
        anchor_ids=train_ids,
        id_label_anchor=id_label_train,
        anchor_feas=pre_feas_train,
        pool_ids=pool_ids,
        id_label_pool=id_label_pool,
        pool_feas=pre_feas_pool,
        neg_dict=neg_dict_train,
    )
    val_dataset = TripletDatasetCross(
        anchor_ids=val_ids,
        id_label_anchor=id_label_val,
        anchor_feas=pre_feas_val,
        pool_ids=pool_ids,
        id_label_pool=id_label_pool,
        pool_feas=pre_feas_pool,
        neg_dict=neg_dict_val,
    )

    train_loader = build_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = build_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    best_val_loss = float('inf')
    patience = 10
    early_stop_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(1, args.epoch + 1):
        if epoch % args.adaptive_rate == 0:
            print(f'==> 第 {epoch} 轮更新训练/验证负样本')
            neg_dict_train = rebuild_negatives(
                model=model,
                anchor_tensor_cpu=pre_feas_tensor_train,
                pool_tensor_cpu=pre_feas_tensor_pool,
                anchor_ids=train_ids,
                pool_ids=pool_ids,
                anchor_labels=labels_train,
                pool_labels=labels_pool,
                device=device,
                knn=args.knn,
                chunk_size=args.mine_chunk_size,
            )
            neg_dict_val = rebuild_negatives(
                model=model,
                anchor_tensor_cpu=pre_feas_tensor_val,
                pool_tensor_cpu=pre_feas_tensor_pool,
                anchor_ids=val_ids,
                pool_ids=pool_ids,
                anchor_labels=labels_val,
                pool_labels=labels_pool,
                device=device,
                knn=args.knn,
                chunk_size=args.mine_chunk_size,
            )

            train_dataset.neg_dict = neg_dict_train
            val_dataset.neg_dict = neg_dict_val

        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            dtype=dtype,
            loss_fn=loss_fn,
            train_mode=True,
            verbose=args.verbose,
            epoch=epoch,
        )
        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            device=device,
            dtype=dtype,
            loss_fn=loss_fn,
            train_mode=False,
            verbose=False,
            epoch=epoch,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            best_path = os.path.join('./result', f'{args.model_name}_best_{timestamp}.pth')
            torch.save(model.state_dict(), best_path)
            print(f'==> 最佳模型已保存到 {best_path}, 验证损失: {val_loss:.4f}, 此时轮数是: {epoch}')
        else:
            early_stop_counter += 1
            print(f'==> 验证损失未提升，early_stop_counter={early_stop_counter}/{patience}')
            if early_stop_counter >= patience and epoch > 40:
                print(f'==> 早停触发！连续 {patience} 轮验证损失无提升，提前结束训练。')
                break

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    plot_losses(train_losses, val_losses,
                save_path=os.path.join('./result', f'{args.model_name}_loss_curve_{timestamp}.png'))
    save_loss_data(train_losses, val_losses, args, save_dir='./result')
    final_path = os.path.join('./result', f'{args.model_name}_final_{timestamp}.pth')
    torch.save(model.state_dict(), final_path)
    print(f'==> 最终模型已保存到 {final_path}')


if __name__ == '__main__':
    main()
