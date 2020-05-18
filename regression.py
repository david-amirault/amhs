import util
from model import *
import numpy as np
import pandas as pd
import statsmodels.api as sm


ids = ['j1s1', 's1t1', 's1a1', 't1s2', 'a1t1', 's2t2', 's2a2', 't2j2', 'a2t2', 'j2j3', 'j2m2', 'j3s3', 'm2n2', 's3t3', 's3a3', 't3s4', 'a3t3', 's4t4', 's4a4', 't4j4', 'a4t4', 'j4j5', 'j4m4', 'j5s5', 'm4n4', 's5t5', 's5a5', 't5s6', 'a5t5', 's6t6', 's6a6', 't6j6', 'a6t6', 'j6j7', 'j6m6', 'j7s7', 'm6n6', 's7t7', 's7a7', 't7s8', 'a7t7', 's8t8', 's8a8', 't8j8', 'a8t8', 'j8j9', 'j8m8', 'j9s9', 'm8n8', 's9t9', 's9a9', 't9s10', 'a9t9', 's10t10', 's10a10', 't10j10', 'a10t10', 'j10m10', 'm10n10', 'm10m11', 'n10n9', 'n10n11', 'n9m9', 'm9m8', 'm9j9', 'n8n7', 'n8n13', 'n7m7', 'm7m6', 'm7j7', 'n6n5', 'n6n15', 'n5m5', 'm5m4', 'm5j5', 'n4n3', 'n4n17', 'n3m3', 'm3m2', 'm3j3', 'n2n1', 'n2n19', 'n1m1', 'm1j1', 'j11s11', 's11t11', 's11a11', 't11s12', 'a11t11', 's12t12', 's12a12', 't12j12', 'a12t12', 'j12j13', 'j12m12', 'j13s13', 'm12n12', 's13t13', 's13a13', 't13s14', 'a13t13', 's14t14', 's14a14', 't14j14', 'a14t14', 'j14j15', 'j14m14', 'j15s15', 'm14n14', 's15t15', 's15a15', 't15s16', 'a15t15', 's16t16', 's16a16', 't16j16', 'a16t16', 'j16j17', 'j16m16', 'j17s17', 'm16n16', 's17t17', 's17a17', 't17s18', 'a17t17', 's18t18', 's18a18', 't18j18', 'a18t18', 'j18j19', 'j18m18', 'j19s19', 'm18n18', 's19t19', 's19a19', 't19s20', 'a19t19', 's20t20', 's20a20', 't20j20', 'a20t20', 'j20m20', 'm20n20', 'm20m1', 'n20n19', 'n20n1', 'n19m19', 'm19m18', 'm19j19', 'n18n17', 'n18n3', 'n17m17', 'm17m16', 'm17j17', 'n16n15', 'n16n5', 'n15m15', 'm15m14', 'm15j15', 'n14n13', 'n14n7', 'n13m13', 'm13m12', 'm13j13', 'n12n11', 'n12n9', 'n11m11', 'm11j11']


def partition(v):
    try:
        i = int(v[-2:])
    except ValueError:
        i = int(v[-1:])
    return i - 1


def main(args, **model_kwargs):
    device = torch.device(args.device)
    adjinit, supports = util.make_graph_inputs(args, device)
    model = GWNet.from_args(args, device, supports, adjinit, **model_kwargs)
    model.load_state_dict(torch.load(args.checkpoint))
    model.to(device)
    model.eval()
    print('torch model loaded successfully')
    data = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size, n_obs=args.n_obs, fill_zeroes=args.fill_zeroes)
    scaler = data['scaler']
    label_data = util.load_dataset(args.label_path, args.batch_size, args.batch_size, args.batch_size, n_obs=args.n_obs, fill_zeroes=args.fill_zeroes)
    print('data loaded successfully')
    y = torch.Tensor(data['y_train']).to(device)
    y = y.transpose(1,3)[:,0,:,:]
    _, yhat = util.calc_tstep_metrics(model, device, data['train_loader'], scaler, y, args.seq_length)
    y = torch.Tensor(label_data['y_train']).to(device)
    y = y.transpose(1,3)[:,0,:,:]
    print('training regression models')
    models = []
    for t in range(args.seq_length):
        ty = y[:,:,t].cpu().numpy()
        tyh = yhat[:,:,t].cpu().numpy()
        ms = []
        for n in range(ty.shape[1]):
            p = partition(ids[n])
            m = sm.OLS(ty[:,n], tyh[:,p:p+1])
            ms.append(m.fit())
        models.append(ms)
        print(t)
    print('evaluating regression models')
    for loader in ['train', 'val', 'test']:
        if loader != 'train':
            y = torch.Tensor(data[f'y_{loader}']).to(device)
            y = y.transpose(1,3)[:,0,:,:]
            _, yhat = util.calc_tstep_metrics(model, device, data[f'{loader}_loader'], scaler, y, args.seq_length)
            y = torch.Tensor(label_data[f'y_{loader}']).to(device)
            y = y.transpose(1,3)[:,0,:,:]
        losses = []
        for t in range(args.seq_length):
            ty = y[:,:,t].cpu().numpy()
            tyh = yhat[:,:,t].cpu().numpy()
            maes = []
            for n in range(ty.shape[1]):
                p = partition(ids[n])
                m = models[t][n]
                mae = np.mean(np.absolute(ty[:,n] - m.predict(tyh[:,p:p+1])))
                maes.append(mae)
            mae = np.mean(mae)
            losses.append(mae)
            print(loader, t, mae)
        print(loader, np.mean(losses))


if __name__ == "__main__":
    parser = util.get_shared_arg_parser()
    parser.add_argument('--label_path', type=str, help='')
    args = parser.parse_args()
    main(args)
