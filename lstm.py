import torch
from torch import nn
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn import metrics
from sklearn.metrics import confusion_matrix


class LSTMClassification(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(LSTMClassification, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        #self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_):
        out, (h, c) = self.lstm(input_)
        logits = self.dropout(out[:,-1,:])
        logits = self.fc(logits)
        logits = self.dropout(logits)
        scores = self.sigmoid(logits)
        return scores


def result_metrics(pred, test, model_name):
        TN, FP, FN, TP = confusion_matrix(test, pred).ravel()
        FPR, TPR, thresholds = metrics.roc_curve(test, pred, pos_label=1)
        AUC = metrics.auc(FPR, TPR)
        # TPR = TP / (TP+FN)
        # FPR = FP / (FP+TN)
        ACC = (TP+TN) / (TP+TN+FP+FN)
        SEN = TP / (TP+FN)
        SPE = TN / (TN+FP)
        BAC = (SEN + SPE) / 2
        print(f'Model: {model_name}')
        print(f'Accuracy: {ACC}')
        print(f'Sensitivity: {SEN}')
        print(f'Specificity: {SPE}')
        print(f'Balanced Accuracy: {BAC}')
        print(f'Auc: {AUC}')
        return ACC, SEN, SPE, BAC, AUC


def train_lstm(x_train, y_train, input_size):
    # train mlp model
    # input_size = 91
    hidden_size = 64
    model = LSTMClassification(input_size, hidden_size)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    model.train()
    epoch = 100
    for epoch in range(1,epoch+1):
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(x_train)
        # Compute Loss
        loss = criterion(y_pred.squeeze(), y_train)
        if epoch % 50 == 0:
            print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
        # Backward pass
        loss.backward()
        optimizer.step()
    return model


def test(x_test, y_test_np, model, model_name):
    model.eval()
    y_pred = model(x_test)
#     after_train = criterion(y_pred.squeeze(), y_test) 
#     print('Test loss after Training' , after_train.item())
#     for pred, true_v in zip(y_pred, y_test_np):
#         print(pred, true_v)
    pred = torch.round(y_pred).cpu().detach().numpy()
    return result_metrics(pred, y_test_np, model_name)


def get_basic_df():
    df = pd.read_excel("data/data.xlsx")
    label = df["HCC"]
    """
    Sex: 0, 1
    HBV: 0, 1
    HIV: 0, 1
    Decompensate cirrhosis: 0, 1
    HCC History: 0, 1
    Transplantation: 0, 1
    ESRD: 0, 1
    Antiviral treatment: 0, 1
    Hepatic decompensation: 0, 0.5, 1
    HCV RNA 6M: 0, 1
    Genotype 1: 0, 1
    DM: 0, 1
    HTN: 0, 1
    Lipid: 0, 1
    F1(M): 0, 1, 2, 3, 4
    F2(M): 0, 1, 2, 3, 4
    """
    one_hot_df = pd.get_dummies(df, columns= ['Sex', 'HBV', 'HIV', 'Decompensate cirrhosis', 
                                 'HCC history', 'Transplantation', 'ESRD',
                                 'Antiviral treatment', 'Hepatic decompensation', 'HCV RNA 6M', 'Genotype 1',
                                 'DM', 'HTN', 'Lipid', 'F1(M)', 'F2(M)'])
    basic_categorial_df = one_hot_df.iloc[:, -39:]
    pre_tx_df = df.iloc[:,26:52]
    baseline_df = df.iloc[:,53:79]
    post_6_month_df = df.iloc[:,80:106]
    post_12_month_df = df.iloc[:,107:133]
    na_idx_6 = post_6_month_df.index[post_6_month_df.isnull().all(1)]
    na_idx_12 = post_12_month_df.index[post_12_month_df.isnull().all(1)]
    for idx in na_idx_6:
        post_6_month_df.iloc[idx,:] = baseline_df.iloc[idx,:]
    for idx in na_idx_12:
        post_12_month_df.iloc[idx,:] = post_6_month_df.iloc[idx,:]
    post_6_month_np = post_6_month_df.to_numpy()
    post_12_month_np = post_12_month_df.to_numpy()
    baseline_np = baseline_df.to_numpy()
    print(post_6_month_np.shape, baseline_np.shape)
    m, n = post_6_month_np.shape
    for i in range(m):
        for j in range(n):
            if math.isnan(post_6_month_np[i,j]):
                post_6_month_np[i,j] = baseline_np[i,j]
    for i in range(m):
        for j in range(n):     
            if type(post_12_month_np[i,j]) == str or math.isnan(post_12_month_np[i,j]):
                post_12_month_np[i,j] = post_6_month_np[i,j]
    post_6_month_parse_df = pd.DataFrame(post_6_month_np)
    post_12_month_parse_df = pd.DataFrame(post_12_month_np)
    return df, label, one_hot_df, basic_categorial_df, pre_tx_df, baseline_df, post_6_month_parse_df, post_12_month_parse_df


def get_time_lstm_df(df, basic_categorial_df, label):
    time_df = pd.concat([df.iloc[:,26:52], basic_categorial_df], axis=1)
    #time_df = basic_categorial_df
    for i in range(30):
        start = i * 27 + 53
        time_df = pd.concat([time_df, df.iloc[:,start:start+26], basic_categorial_df], axis=1)
        #time_df = pd.concat([time_df, basic_categorial_df], axis=1)
    #time_df = pd.concat([time_df, df.iloc[:,-39:]], axis=1)
    time_df = pd.concat([time_df, label], axis=1)
    time_df = time_df.replace(np.nan, 0)
    time_df = time_df.replace(r'^\s*$', 0, regex=True)
    df['HCC year'] = df['HCC year'].replace(r'^\s*$', 0, regex=True)
    hcc_year = df['HCC year'].replace(np.nan, 0)
    for i, year in enumerate(hcc_year):
        no = int(float(year)*2)
        if no != 0:
            time_df.iloc[i,(no+1)*65:(no+2)*65] = 0
    return time_df


def main():
    df, label, one_hot_df, basic_categorial_df, pre_tx_df, baseline_df, post_6_month_df, post_12_month_df = get_basic_df()
    time_df = get_time_lstm_df(df, basic_categorial_df, label)
    kf = KFold()
    accs, sens, spes, bacs = [], [], [], []
    normalization = True
    time_df = time_df.sample(frac=1).reset_index(drop=True)
    for train_index, test_index in kf.split(time_df):
        for idx in test_index:
            assert idx not in train_index
        train_df, test_df = time_df.iloc[train_index,:], time_df.iloc[test_index,:]
        train_df = train_df.append([train_df[train_df['HCC'] == 1]] * 10, ignore_index=True)
        print(train_df['HCC'].value_counts())
        print(test_df['HCC'].value_counts())
        x_train_np = train_df.iloc[:,:-1].to_numpy().astype(float)
        y_train_np = train_df.iloc[:,-1].to_numpy().astype(float)
        x_test_np = test_df.iloc[:,:-1].to_numpy().astype(float)
        y_test_np = test_df.iloc[:,-1].to_numpy().astype(float)
        if normalization:
            x_train_np = normalize(x_train_np, axis=0)
            x_test_np = normalize(x_test_np, axis=0)
        print(x_train_np.shape, y_train_np.shape, x_test_np.shape, y_test_np.shape)
        x_train_np_lstm = x_train_np.reshape(-1, 31, 65)
        x_test_np_lstm = x_test_np.reshape(-1, 31, 65)
        x_train = torch.from_numpy(x_train_np_lstm).float()
        y_train = torch.from_numpy(y_train_np).float()
        x_test = torch.from_numpy(x_test_np_lstm).float()
        y_test = torch.from_numpy(y_test_np).float()
        model = train_lstm(x_train, y_train, 65)
        ACC, SEN, SPE, BAC, AUC = test(x_test, y_test, model, 'lstm')
        test(x_train, y_train, model, 'lstm')
        accs.append(ACC)
        sens.append(SEN)
        spes.append(SPE)
        bacs.append(BAC)

    print('5-fold cross validation result:')
    print(f'ACC: {sum(accs) / 5}') 
    print(f'SEN: {sum(sens) / 5}')
    print(f'SPE: {sum(spes) / 5}')
    print(f'BAC: {sum(bacs) / 5}')


if __name__ == "__main__":
    main()