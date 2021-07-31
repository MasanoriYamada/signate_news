# -*- coding: utf-8 -*-
import io
import os
import pickle

import lightgbm as lgbm
import numpy as np
import pandas as pd
import talib as ta
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm

TRAIN_START = "2016-02-01"  # テクニカルを行うために１ヶ月のbuffer
TRAIN_END = "2019-12-30"
VAL_START = "2020-02-01"
VAL_END = "2020-11-30"
TEST_START = '2021-02-01'
TEST_END = '2021-02-05'


class ScoringService(object):
    models = None
    @classmethod
    def get_model(cls, model_path="../model", labels=None):
        print('read checkpoints')
        path = os.path.join(model_path, f"checkpoints.pickle")
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        cls.models = checkpoint['models']
        cls.keys = checkpoint['keys']
        assert len(cls.keys) == len(cls.models)
        return True

    @classmethod
    def predict(cls, inputs):
        if "purchase_date" in inputs.keys():
            # purchase_dateの最も古い日付を設定
            purchase_df = pd.read_csv(inputs["purchase_date"])
            TEST_START = purchase_df.sort_values("Purchase Date").iloc[0, 0]
            print(f"update TEST_START: {TEST_START}")
        # pred date
        PRED_DATE = pd.Timestamp(TEST_START) - pd.Timedelta("3D")

        # load data
        df_price = cls.load_data(inputs)

        # 特徴量ベクトルを生成
        print('make feature vector')
        X = cls.creaate_feature(df_price)
        y = cls.to_target(df_price)
        y[y.isin([np.inf, -np.inf])] = 0.0  # inf 防止
        print('predict')

        pred = cls.pred_lgbm(X.loc[PRED_DATE], y.loc[PRED_DATE], cls.models['1'], cls.keys['1'])
        pf_df = X.loc[PRED_DATE].copy()
        pf_df['ret'] = pred

        # make portfolio
        print('make portfolio')
        df_portfolio = cls.make_portfolio(pf_df, PRED_DATE, TEST_START)

        out = io.StringIO()
        df_portfolio.to_csv(out, header=True)
        #df_portfolio.to_csv('../../submit.csv', header=True)
        return out.getvalue()

    @classmethod
    def train(cls, inputs):
        # load data
        df_price = cls.load_data(inputs)

        print('make feature vector')
        X = cls.creaate_feature(df_price)
        y = cls.to_target(df_price)

        print('train')
        def get_index(input_df, start, end):
            return input_df.reset_index().reset_index().set_index('EndOfDayQuote Date').loc[start:end, 'index'].values

        idx_train = get_index(df_price, TRAIN_START, TRAIN_END)
        idx_valid = get_index(df_price, VAL_START, VAL_END)
        cv = [(idx_train, idx_valid)]
        params = {
            'objective': 'rmse',
            'learning_rate': .1,
            'reg_lambda': 1.,
            'reg_alpha': .1,
            'max_depth': 5,
            'n_estimators': 10000,
            'colsample_bytree': .5,
            'min_child_samples': 10,
            'subsample_freq': 3,
            'subsample': .9,
            'importance_type': 'gain',
            'random_state': 71,
        }
        oof, models = cls.fit_lgbm(X.values, y.values.reshape(-1), cv=cv, params=params, verbose=500)
        cls.models = models
        print('save models')
        checkpoint = {'keys': [list(X.keys())], 'models': models}
        import pickle
        with open('../model/checkpoints.pickle', 'wb') as f:
            pickle.dump(checkpoint, f)
        return models

    @classmethod
    def load_data(cls, inputs):
        df_stock_list = pd.read_csv(inputs["stock_list"])
        codes = df_stock_list.loc[
            df_stock_list.loc[:, "universe_comp2"] == True, "Local Code"
        ].unique()
        df_price = pd.read_csv(inputs["stock_price"]).set_index("EndOfDayQuote Date")
        df_price.index = pd.to_datetime(df_price.index, format="%Y-%m-%d")
        df_price = df_price.reset_index().sort_values(['Local Code', 'EndOfDayQuote Date']).set_index(
            'EndOfDayQuote Date')
        # reduce data
        start_dt = pd.Timestamp(TRAIN_START)
        pred_start_dt = pd.Timestamp(start_dt) - pd.Timedelta("3D")
        n = 30
        data_start_dt = pred_start_dt - pd.offsets.BDay(n)
        filter_date = df_price.index >= data_start_dt
        filter_universe = df_price.loc[:, "Local Code"].isin(codes)
        df_price = df_price.loc[filter_date & filter_universe]
        return df_price

    @classmethod
    def make_portfolio(cls, pf_df, PRED_DATE, TEST_START):
        top_k = 9
        df_portfolio = pd.DataFrame()
        df_portfolio['Local Code'] = pf_df.loc[PRED_DATE].sort_values(['ret']).tail(top_k)['Local Code']
        df_portfolio['date'] = TEST_START
        df_portfolio['budget'] = int(1000000 / top_k)
        return df_portfolio

    @classmethod
    def creaate_feature(cls, input_df):
        def create_numeric_feature(input_df):
            all_columns = set(input_df.keys())
            except_columns = set(['EndOfDayQuote CumulativeAdjustmentFactor', 'EndOfDayQuote PreviousCloseDate', 'EndOfDayQuote PreviousExchangeOfficialCloseDate'])
            use_columns = all_columns - except_columns
            return input_df[use_columns].copy()

        def create_talib_feature(input_df):
            out_df = []
            target_label = 'EndOfDayQuote ExchangeOfficialClose'
            for target_code in tqdm(input_df['Local Code'].unique()):
                ret = {}
                df = input_df.loc[input_df.loc[:, 'Local Code']==target_code, target_label]
                prices = np.array(df, dtype='f8')
                ret['rsi7'] = ta.RSI(prices, timeperiod=7)
                ret['sma14'] = ta.SMA(prices, timeperiod=14)
                ret['sma7'] = ta.SMA(prices, timeperiod=7)
                ret['bb_up'], ret['bb_mid'], ret['bb_low'] = ta.BBANDS(prices, timeperiod=14)
                ret['mom'] = ta.MOM(prices, timeperiod=14)
                tmp_df = pd.DataFrame(ret, index=df.index)
                ret['sma7_diff'] = tmp_df['sma7'].diff()
                ret['sma14_diff'] = tmp_df['sma14'].diff()
                ret['sma_diff'] =tmp_df['sma7'] -tmp_df['sma14']
                ret['bb_diff'] = tmp_df['bb_mid'] - tmp_df['bb_up']
                code_df = pd.DataFrame(ret, index=df.index)
                out_df.append(code_df)
            return pd.concat(out_df, axis=0)

        def create_return_feature(input_df):
            out_df = []
            target_label = 'EndOfDayQuote ExchangeOfficialClose'
            for target_code in tqdm(input_df['Local Code'].unique()):
                ret = {}
                df = input_df.loc[input_df.loc[:, 'Local Code'] == target_code, target_label]
                ret['retrun1'] = input_df[input_df['Local Code'] == target_code][target_label].pct_change(periods=1)
                ret['retrun7'] = input_df[input_df['Local Code'] == target_code][target_label].pct_change(periods=7)
                ret['retrun14'] = input_df[input_df['Local Code'] == target_code][target_label].pct_change(periods=14)
                code_df = pd.DataFrame(ret, index=df.index)
                out_df.append(code_df)
            return pd.concat(out_df, axis=0)

        def to_feature(input_df):
            """input_df を特徴量行列に変換した新しいデータフレームを返す.
            """
            processors = [
                create_numeric_feature,
                create_talib_feature,
                create_return_feature
            ]

            out_df = pd.DataFrame()

            for func in tqdm(processors, total=len(processors)):
                _df = func(input_df)
                # 長さが等しいことをチェック (ずれている場合, func の実装がおかしい)
                assert len(_df) == len(input_df), func.__name__
                out_df = pd.concat([out_df, _df], axis=1)
            return out_df

        X = to_feature(input_df)
        new_X = X.groupby('Local Code').fillna(method='bfill')  # 過去の情報から作られる
        new_X['Local Code'] = X['Local Code']
        X = new_X
        return X

    @classmethod
    def to_target(cls, input_df):
        df_copy = input_df.copy()
        out_df = []
        end_label = 'EndOfDayQuote ExchangeOfficialClose'
        start_label = 'EndOfDayQuote Open'
        # open が存在しない場合の0割を回避
        df_copy.loc[df_copy['EndOfDayQuote Open'] == 0, 'EndOfDayQuote Open'] = df_copy.loc[
            df_copy['EndOfDayQuote Open'] == 0, 'EndOfDayQuote PreviousClose']
        df_copy.loc[df_copy['EndOfDayQuote Open'] == 0, 'EndOfDayQuote Open'] = df_copy.loc[
            df_copy['EndOfDayQuote Open'] == 0, 'EndOfDayQuote ExchangeOfficialClose']
        df_copy.loc[df_copy['EndOfDayQuote Open'] == 0, 'EndOfDayQuote Open'] = 1.0
        # print(df_copy.sort_values(['EndOfDayQuote Open']))
        for target_code in tqdm(df_copy['Local Code'].unique()):
            ret = {}
            df = input_df.loc[df_copy.loc[:, 'Local Code'] == target_code]
            ret['target'] = (df[end_label].shift(-6) - df[start_label].shift(-1)) / df[start_label].shift(-1)
            code_df = pd.DataFrame(ret, index=df.index)
            out_df.append(code_df)
        return pd.concat(out_df, axis=0)

    @classmethod
    def fit_lgbm(cls, X, y, cv, params: dict = None, verbose: int = 50):
        """lightGBM with CrossValidation"""

        # パラメータがないときは、空の dict で置き換える
        if params is None:
            params = {}

        models = []
        # training data の target と同じだけのゼロ配列を用意
        oof_pred = np.zeros_like(y, dtype=np.float)

        for i, (idx_train, idx_valid) in enumerate(cv):
            # この部分が交差検証のところです。データセットを cv instance によって分割します
            # training data を trian/valid に分割
            x_train, y_train = X[idx_train], y[idx_train]
            x_valid, y_valid = X[idx_valid], y[idx_valid]

            clf = lgbm.LGBMRegressor(**params)

            clf.fit(x_train, y_train,
                    eval_set=[(x_valid, y_valid)],
                    early_stopping_rounds=100,
                    verbose=verbose)

            pred_i = clf.predict(x_valid)
            oof_pred[idx_valid] = pred_i
            models.append(clf)
            # print(f'Fold {i} RMSLE: {mean_squared_error(y_valid, pred_i) ** .5:.4f}')

        score = mean_squared_error(y, oof_pred) ** .5
        print('-' * 50)
        print('FINISHED | Whole RMSLE: {:.4f}'.format(score))
        return oof_pred, models

    @classmethod
    def pred_lgbm(cls, X, y, model, key):
        scores = []
        preds = []
        length = len(X)
        for model_, key_ in zip(model, key):
            y = y.values
            X = X[key_].values
            pred = model_.predict(X)
            #score = mean_squared_error(y, pred) ** .5 / length
            preds.append(pred)
            #scores.append(score)
        #print('score', score)
        print('-' * 50)
        pred = np.vstack(preds).mean(axis=0)
        #socre = np.vstack(scores).mean(axis=0)
        #print('FINISHED | Whole RMSLE: {:.4f}'.format(score))
        return pred#, score


if __name__ == '__main__':
    dataset_dir = "../../data"
    inputs = {
        "stock_list": f"{dataset_dir}/stock_list.csv.gz",
        "stock_price": f"{dataset_dir}/stock_price.csv.gz",
        "stock_fin": f"{dataset_dir}/stock_fin.csv.gz",
        "stock_fin_price": f"{dataset_dir}/stock_fin_price.csv.gz",
        # ニュースデータ
        "tdnet": f"{dataset_dir}/tdnet.csv.gz",
        "disclosureItems": f"{dataset_dir}/disclosureItems.csv.gz",
        "nikkei_article": f"{dataset_dir}/nikkei_article.csv.gz",
        "article": f"{dataset_dir}/article.csv.gz",
        "industry": f"{dataset_dir}/industry.csv.gz",
        "industry2": f"{dataset_dir}/industry2.csv.gz",
        "region": f"{dataset_dir}/region.csv.gz",
        "theme": f"{dataset_dir}/theme.csv.gz",
        # 目的変数データ
        "stock_labels": f"{dataset_dir}/stock_labels.csv.gz",
        # 日付
        "purchase_date": f"{dataset_dir}/purchase_date.csv"
    }

    # train
    #ScoringService.train(inputs)

    # pred
    if ScoringService.get_model():
        out = ScoringService.predict(inputs)
        print(out)
    else:
        print('load error')
