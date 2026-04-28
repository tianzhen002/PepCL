import argparse
import datetime
import pickle
import sys
import time
import random
import numpy as np
import os

from torch.utils.data import DataLoader, Dataset
from torchnet import meter
import torch
from torch import nn

from preprocess import parse_fasta, get_pretrained_features
from valid_metrices import CFM_eval_metrics, print_results, eval_metrics, th_eval_metrics

import warnings

warnings.filterwarnings("ignore", message="TypedStorage is deprecated.")
warnings.filterwarnings("ignore", message="indexing with dtype torch.uint8 is now deprecated")


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(RotaryPositionalEncoding, self).__init__()
        assert d_model % 2 == 0, "d_model must be even for rotary encoding"
        self.d_model = d_model
        self.register_buffer('angles', self._precompute_angles(max_len, d_model))

    def _precompute_angles(self, max_len, d_model):
        half_dim = d_model // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        angles = positions * inv_freq.unsqueeze(0)
        return angles

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        half = d_model // 2
        angles = self.angles[:seq_len, :].unsqueeze(0).expand(batch, -1, -1)
        x1 = x[:, :, :half]
        x2 = x[:, :, half:]
        x1_rot = x1 * torch.cos(angles) - x2 * torch.sin(angles)
        x2_rot = x1 * torch.sin(angles) + x2 * torch.cos(angles)
        x_rot = torch.cat([x1_rot, x2_rot], dim=-1)
        return x_rot


class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, max_len=5000):
        super(TransformerModel, self).__init__()
        self.pos_encoder = RotaryPositionalEncoding(d_model, max_len=max_len)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward
        )

    def forward(self, src, tgt):
        src = self.pos_encoder(src)
        output = self.transformer(src, tgt)
        return output


class OnlyDuibiAIMP(torch.nn.Module):
    """
    Standalone only_duibi model.

    Important:
    - The module names intentionally match the original AIMP as much as possible.
    - Some modules (e.g. pre_embedding) are kept even though forward() does not use them,
      so that old only_duibi checkpoints from the ablation script stay easier to compare/load
      and initialization order stays aligned with the original model definition.
    """
    def __init__(self, pre_feas_dim, hidden, n_transformer, dropout):
        super(OnlyDuibiAIMP, self).__init__()

        self.n_transformer = n_transformer

        # keep this branch for checkpoint / init-order compatibility
        self.pre_embedding = nn.Sequential(
            nn.Conv1d(pre_feas_dim, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=1),
        )

        self.dbemb = nn.Sequential(
            nn.Linear(128, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 128),
            nn.ReLU()
        )

        # only_duibi path: fusion_in_dim = 128
        self.embedding = nn.Sequential(
            nn.Conv1d(128, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=1),
        )

        # keep both BN slots to match original naming/layout
        self.bn = nn.ModuleList([
            nn.BatchNorm1d(pre_feas_dim),
            nn.BatchNorm1d(128)
        ])

        self.transformer = TransformerModel(
            d_model=hidden,
            nhead=4,
            num_layers=self.n_transformer,
            dim_feedforward=2048
        )

        self.transformer_act = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=1),
        )

        self.transformer_res = nn.Sequential(
            nn.Conv1d(hidden + hidden, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=1),
        )

        self.transformer_pool = nn.AdaptiveAvgPool2d((1, None))

        self.clf = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, pre_feas=None, duibi=None):
        if duibi is None:
            raise ValueError("duibi is required in OnlyDuibiAIMP")

        duibi = self.bn[1](duibi.permute(0, 2, 1)).permute(0, 2, 1)
        N, T, C = duibi.shape
        duibi_feat_out = self.dbemb(duibi.reshape(-1, C)).reshape(N, T, 128)

        feas_em = self.embedding(duibi_feat_out.permute(0, 2, 1)).permute(0, 2, 1)

        transformer_out = self.transformer(feas_em, feas_em)
        transformer_out = self.transformer_act(transformer_out.permute(0, 2, 1)).permute(0, 2, 1)
        transformer_out = self.transformer_res(
            torch.cat([transformer_out, feas_em], dim=-1).permute(0, 2, 1)
        ).permute(0, 2, 1)

        transformer_out = self.transformer_pool(transformer_out).squeeze(1)

        out = self.clf(transformer_out)
        out = torch.nn.functional.softmax(out, -1)
        return out[:, -1]


class LayerNormNet(nn.Module):
    def __init__(self, hidden_dim, out_dim, device, dtype, drop_out=0.5):
        super(LayerNormNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.drop_out = drop_out
        self.device = device
        self.dtype = dtype

        self.fc1 = nn.Linear(1024, hidden_dim, dtype=dtype, device=device)
        self.ln1 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=dtype, device=device)
        self.ln2 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc3 = nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)

        self.dropout1 = nn.Dropout(p=drop_out)
        self.dropout2 = nn.Dropout(p=drop_out)

    def forward(self, x):
        N, theta, _ = x.shape
        x = x.view(N * theta, -1)
        x = self.dropout1(self.ln1(self.fc1(x)))
        x = torch.relu(x)
        x = self.dropout2(self.ln2(self.fc2(x)))
        x = torch.relu(x)
        x = self.fc3(x)
        x = x.view(N, theta, -1)
        return x


# =========================
#  training utilities
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Standalone training script for only_duibi mode.")
    parser.add_argument("--type", "-type", dest="type", type=str, default='AMP'
                    )
    parser.add_argument("--train_fasta", "-train_fasta", dest='train_fasta', type=str, default='train.txt')
    parser.add_argument("--valid_fasta", "-valid_fasta", dest='valid_fasta', type=str, default='valid.txt')
    parser.add_argument("--test_fasta", "-test_fasta", dest='test_fasta', type=str, default='test.txt')
    parser.add_argument("--hidden", "-hidden", dest='hidden', type=int, default=256)
    parser.add_argument("--drop", "-drop", dest='drop', type=float, default=0.5)
    parser.add_argument("--n_transformer", "-n_transformer", dest='n_transformer', type=int, default=1)
    parser.add_argument("--lr", "-lr", dest='lr', type=float, default=0.0001)
    parser.add_argument("--batch_size", "-batch_size", dest='batch_size', type=int, default=256)
    parser.add_argument("--seed", "-seed", dest='seed', type=int, default=1999)
    parser.add_argument("--epoch", "-epoch", dest='epoch', type=int, default=100)
    parser.add_argument("--duibi_ckpt", dest="duibi_ckpt", type=str, default="split10_triplet_best.pth",
                        help="Path to frozen contrastive model checkpoint")
    parser.add_argument("--fix_seed", action="store_true",
                        help="Enable deterministic seed setup. Default is off to stay closer to the old ablation script.")
    return parser.parse_args()


def checkargs(args):
    if args.type is None or args.train_fasta is None or args.test_fasta is None:
        print('ERROR: please input the necessary parameters!')
        raise ValueError

    if args.type not in ['AMP', 'AIP']:
        print(f'ERROR: type "{args.type}" is not supported!')
        raise ValueError


class Config:
    def __init__(self, args):
        self.type = args.type
        self.Dataset_dir = f'../datasets/{self.type}'
        self.train_fasta = args.train_fasta
        self.valid_fasta = args.valid_fasta
        self.test_fasta = args.test_fasta
        self.batch_size = args.batch_size
        self.hidden = args.hidden
        self.n_transformer = args.n_transformer
        self.drop = args.drop
        self.epoch = args.epoch
        self.lr = args.lr
        self.seed = args.seed
        self.duibi_ckpt = args.duibi_ckpt
        self.fix_seed = args.fix_seed

        self.feature_path = f'{self.Dataset_dir}/feature'
        self.checkpoints = f'{self.Dataset_dir}/checkpoints'
        self.model_time = None
        self.train = True

        localtime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        self.model_path = f'{self.checkpoints}/{localtime}_only_duibi_standalone'
        self.submodel_path = self.model_path + '/model'
        self.sublog_path = self.model_path + '/log'

        if not os.path.exists(self.submodel_path):
            os.makedirs(self.submodel_path)
        if not os.path.exists(self.sublog_path):
            os.makedirs(self.sublog_path)

        self.max_metric = 'PRC'
        self.theta = 40
        self.saved_model_num = 3
        self.early_stop_epochs = 20

    def print_config(self):
        for name, value in vars(self).items():
            print(f'{name} = {value}')


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, 'ab', buffering=0)

    def write(self, message):
        self.terminal.write(message)
        try:
            self.log.write(message.encode('utf-8'))
        except ValueError:
            pass

    def close(self):
        self.log.close()
        sys.stdout = self.terminal

    def flush(self):
        pass


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def data_pre(fasta_file, feature_path, theta=40):
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)

    fasta_file_path = f'{opt.Dataset_dir}/{fasta_file}'
    names, sequences, labels = parse_fasta(fasta_file_path, number=None)

    name = fasta_file.split('.')[0]
    pre_feas = get_pretrained_features(names, sequences, f'{feature_path}/{name}.h5', theta=theta)

    return pre_feas, labels, names, sequences


class myDataset(Dataset):
    def __init__(self, pre_feas, duibi, labels):
        self.pre_feas = pre_feas
        self.duibi = duibi
        self.labels = labels

    def __getitem__(self, index):
        return self.pre_feas[index], self.duibi[index], self.labels[index]

    def __len__(self):
        return self.labels.size(0)


def train(opt, device, model, train_data, valid_data, test_data):
    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=opt.early_stop_epochs // 2,
        min_lr=1e-6
    )
    criterion = torch.nn.BCELoss()

    model.to(device)
    criterion.to(device)

    loss_meter = meter.AverageValueMeter()
    early_stop_iter = 0
    max_metric_val = -1
    nsave_model = 0

    max_test_acc = -1
    best_test_model_state = None
    best_test_th = None

    begintime = datetime.datetime.now()
    print('Time:', begintime)

    for epoch in range(opt.epoch):
        nstep = len(train_dataloader)

        for _, data in enumerate(train_dataloader):
            model.train()
            pre_feas, duibi, target = data
            pre_feas, duibi, target = pre_feas.to(device), duibi.to(device), target.to(device)

            optimizer.zero_grad()
            score = model(pre_feas, duibi)
            target = target.float()
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.item())

        nowtime = datetime.datetime.now()
        print('|| Epoch{} step{} || lr={:.6f} | train_loss={:.5f}'.format(
            epoch, nstep, optimizer.param_groups[0]['lr'], loss_meter.mean
        ))
        print('Time:', nowtime)
        print('Timedelta: %s seconds' % (nowtime - begintime).seconds)

        val_th, val_acc, val_rec, val_pre, val_F1, val_spe, val_mcc, val_auc, val_prc = val(
            opt, device, model, valid_data, 'valid', val_th=None
        )

        test_th, test_acc, test_rec, test_pre, test_F1, test_spe, test_mcc, test_auc, test_prc = val(
            opt, device, model, test_data, 'test', val_th
        )

        if test_acc > max_test_acc:
            max_test_acc = test_acc
            best_test_model_state = model.state_dict()
            best_test_th = test_th

        if opt.max_metric == 'AUC':
            metrice_val = val_auc
        elif opt.max_metric == 'MCC':
            metrice_val = val_mcc
        elif opt.max_metric == 'F1':
            metrice_val = val_F1
        elif opt.max_metric == 'ACC':
            metrice_val = val_acc
        elif opt.max_metric == 'PRC':
            metrice_val = val_prc
        else:
            raise ValueError('ERROR: opt.max_metric.')

        if metrice_val > max_metric_val:
            max_metric_val = metrice_val
            if nsave_model < opt.saved_model_num:
                save_path = '{}/model{}.pth'.format(opt.submodel_path, nsave_model)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'val_th': val_th,
                }, save_path)
                nsave_model += 1
            else:
                save_path = '{}/model{}.pth'.format(opt.submodel_path, nsave_model - 1)
                for model_i in range(1, opt.saved_model_num):
                    os.system('mv {}/model{}.pth {}/model{}.pth'.format(
                        opt.submodel_path, model_i, opt.submodel_path, model_i - 1
                    ))
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'val_th': val_th,
                }, save_path)
            early_stop_iter = 0
        else:
            early_stop_iter += 1
            if early_stop_iter == opt.early_stop_epochs:
                break

        scheduler.step(metrice_val)
        loss_meter.reset()

    if best_test_model_state is not None:
        test_save_path = '{}/best_test_model.pth'.format(opt.submodel_path)
        torch.save({
            'model_state_dict': best_test_model_state,
            'test_th': best_test_th,
            'test_acc': max_test_acc,
        }, test_save_path)
        print('Saved best test model with ACC: {:.4f} at {}'.format(max_test_acc, test_save_path))


def val(opt, device, model, valid_data, dataset_type, val_th=None):
    valid_dataloader = DataLoader(valid_data, batch_size=2 * opt.batch_size, shuffle=False, pin_memory=False)
    model.eval()

    if val_th is not None:
        AUC_meter = meter.AUCMeter()
        PRC_meter = meter.APMeter()
        Confusion_meter = meter.ConfusionMeter(k=2)

        with torch.no_grad():
            for _, data in enumerate(valid_dataloader):
                pre_feas, duibi, target = data
                pre_feas, duibi, target = pre_feas.to(device), duibi.to(device), target.to(device)

                score = model(pre_feas, duibi).float()
                AUC_meter.add(score, target)
                PRC_meter.add(score, target)

                pred_bi = target.data.new(score.shape).fill_(0)
                pred_bi[score > val_th] = 1
                Confusion_meter.add(pred_bi, target)

        val_auc = AUC_meter.value()[0]
        val_prc = PRC_meter.value().item()
        cfm = Confusion_meter.value()
        val_acc, val_rec, val_pre, val_F1, val_spe, val_mcc = CFM_eval_metrics(cfm)

        print(
            '{} result: th={:.2f} acc={:.3f} sen={:.3f} pre={:.3f} F1={:.3f}, spe={:.3f} MCC={:.3f} AUC={:.3f} PRC={:.3f}'
            .format(dataset_type, val_th, val_acc, val_rec, val_pre, val_F1, val_spe, val_mcc, val_auc, val_prc)
        )
    else:
        AUC_meter = meter.AUCMeter()
        PRC_meter = meter.APMeter()
        for j in range(2, 100, 2):
            th = j / 100.0
            locals()['Confusion_meter_' + str(th)] = meter.ConfusionMeter(k=2)

        with torch.no_grad():
            for _, data in enumerate(valid_dataloader):
                pre_feas, duibi, target = data
                pre_feas, duibi, target = pre_feas.to(device), duibi.to(device), target.to(device)
                target = target.float()

                score = model(pre_feas, duibi).float()
                AUC_meter.add(score, target)
                PRC_meter.add(score, target)

                for j in range(2, 100, 2):
                    th = j / 100.0
                    pred_bi = target.data.new(score.shape).fill_(0)
                    pred_bi[score > th] = 1
                    locals()['Confusion_meter_' + str(th)].add(pred_bi, target)

        val_auc = AUC_meter.value()[0]
        val_prc = PRC_meter.value().item()
        val_acc, val_rec, val_pre, val_F1, val_spe, val_mcc = -1, -1, -1, -1, -1, -1

        for j in range(2, 100, 2):
            th = j / 100.0
            cfm = locals()['Confusion_meter_' + str(th)].value()
            acc, rec, pre, F1, spe, mcc = CFM_eval_metrics(cfm)
            if F1 > val_F1:
                val_acc, val_rec, val_pre, val_F1, val_spe, val_mcc = acc, rec, pre, F1, spe, mcc
                val_th = th

        print(
            '{} result: th={:.2f} acc={:.3f} sen={:.3f} pre={:.3f} F1={:.3f}, spe={:.3f} MCC={:.3f} AUC={:.3f} PRC={:.3f}'
            .format(dataset_type, val_th, val_acc, val_rec, val_pre, val_F1, val_spe, val_mcc, val_auc, val_prc)
        )

    return val_th, val_acc, val_rec, val_pre, val_F1, val_spe, val_mcc, val_auc, val_prc


def test(opt, device, model, test_data):
    avg_test_probs = []
    avg_test_targets = []

    for model_i in range(opt.saved_model_num):
        model_path = '{}/model{}.pth'.format(opt.submodel_path, model_i)
        checkpoints = torch.load(model_path)
        model.load_state_dict(checkpoints['model_state_dict'])
        model.to(device)
        model.eval()

        test_dataloader = DataLoader(test_data, batch_size=2 * opt.batch_size, shuffle=False, pin_memory=False)
        test_probs = []
        test_targets = []

        with torch.no_grad():
            for _, data in enumerate(test_dataloader):
                pre_feas, duibi, target = data
                pre_feas, duibi, target = pre_feas.to(device), duibi.to(device), target.to(device)
                target = target.float()
                score = model(pre_feas, duibi).float()
                test_probs += score.tolist()
                test_targets += target.tolist()

        test_probs = np.array(test_probs)
        test_targets = np.array(test_targets)
        avg_test_probs.append(test_probs.reshape(-1, 1))
        avg_test_targets.append(test_targets.reshape(-1, 1))

    avg_test_probs = np.concatenate(avg_test_probs, axis=1)
    avg_test_probs = np.average(avg_test_probs, axis=1)

    avg_test_targets = np.concatenate(avg_test_targets, axis=1)
    avg_test_targets = np.average(avg_test_targets, axis=1)

    return avg_test_probs, avg_test_targets


def build_duibi_features(opt, pre_feas_train, pre_feas_valid, pre_feas_test, device):
    modelduibi = LayerNormNet(
        hidden_dim=256,
        out_dim=128,
        device=device,
        dtype=torch.float32,
        drop_out=0.5
    )

    ckpt = torch.load(opt.duibi_ckpt, weights_only=True)
    modelduibi.load_state_dict(ckpt)
    modelduibi = modelduibi.to(device)
    modelduibi.eval()

    with torch.no_grad():
        duibi_train = modelduibi(pre_feas_train)
        duibi_valid = modelduibi(pre_feas_valid)
        duibi_test = modelduibi(pre_feas_test)

    return duibi_train, duibi_valid, duibi_test


def main(opt, device):
    if opt.fix_seed:
        seed_everything(opt.seed)

    with open(f'{opt.model_path}/params.pkl', 'wb') as f:
        pickle.dump(opt, f)

    feature_path = opt.feature_path
    print('Loading the data...')

    pre_feas_train, label_train, names_train, seqs_train = data_pre(
        fasta_file=opt.train_fasta, feature_path=feature_path, theta=opt.theta
    )
    pre_feas_valid, label_valid, names_valid, seqs_valid = data_pre(
        fasta_file=opt.valid_fasta, feature_path=feature_path, theta=opt.theta
    )
    pre_feas_test, label_test, names_test, seqs_test = data_pre(
        fasta_file=opt.test_fasta, feature_path=feature_path, theta=opt.theta
    )

    pre_feas_train = torch.Tensor(pre_feas_train).to(device)
    pre_feas_valid = torch.Tensor(pre_feas_valid).to(device)
    pre_feas_test = torch.Tensor(pre_feas_test).to(device)

    duibi_train, duibi_valid, duibi_test = build_duibi_features(
        opt, pre_feas_train, pre_feas_valid, pre_feas_test, device
    )

    print('Finish loading data!')

    label_train = torch.Tensor(label_train).long()
    label_valid = torch.Tensor(label_valid).long()
    label_test = torch.Tensor(label_test).long()

    train_data = myDataset(pre_feas_train, duibi_train, label_train)
    valid_data = myDataset(pre_feas_valid, duibi_valid, label_valid)
    test_data = myDataset(pre_feas_test, duibi_test, label_test)

    print('Loading the model...')
    model = OnlyDuibiAIMP(
        pre_feas_dim=1024,
        hidden=opt.hidden,
        n_transformer=opt.n_transformer,
        dropout=opt.drop,
    )

    print("Train...")
    train(opt, device, model, train_data, valid_data, test_data)
    print('Training is finished!')

    print('Test...')
    valid_probs, valid_labels = test(opt, device, model, valid_data)
    test_probs, test_labels = test(opt, device, model, test_data)

    th_, acc_, rec_, pre_, f1_, spe_, mcc_, roc_, prc_, pred_class = eval_metrics(valid_probs, valid_labels)
    valid_matrices = th_, acc_, rec_, pre_, f1_, spe_, mcc_, roc_, prc_

    th_, acc_, rec_, pre_, f1_, spe_, mcc_, roc_, prc_, pred_class = th_eval_metrics(th_, test_probs, test_labels)
    test_matrices = th_, acc_, rec_, pre_, f1_, spe_, mcc_, roc_, prc_

    print_results(valid_matrices, test_matrices)

    results = {
        'valid_probs': valid_probs,
        'valid_labels': valid_labels,
        'test_probs': test_probs,
        'test_labels': test_labels,
        'feature_mode': 'only_duibi_standalone'
    }

    with open(opt.sublog_path + '/results.pkl', 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    arguments = parse_args()
    checkargs(arguments)
    opt = Config(arguments)
    sys.stdout = Logger(opt.model_path + '/training.log')
    opt.print_config()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    main(opt, device)

    sys.stdout.log.close()
