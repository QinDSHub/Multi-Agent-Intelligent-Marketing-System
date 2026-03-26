#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os,gc, math, json, argparse
import numpy as np
from datetime import datetime
from natsort import natsorted
from sklearn.model_selection import StratifiedShuffleSplit

def stat_feat(path):
    # extract basic info of users
    veh_df = pd.read_csv(os.path.join(path,'vehicle3.csv'))
    cols = ['VIN','车主性质','车型','family_name']
    veh_df = veh_df[cols]
    veh_df.columns = ['VIN','owner_type','car_mode','car_level']
    
    # extract member feat
    member_info = pd.read_csv(os.path.join(path, 'member_info.csv'))
    member_info = member_info[['VIN','会员等级']].drop_duplicates()
    member_info = member_info[~member_info['会员等级'].isna()].reset_index(drop=True)
    member_info.columns = ['VIN','member_level']
    
    df = veh_df.merge(member_info, on=['VIN'], how='outer').drop_duplicates().reset_index(drop=True)
    
    # fill up missing value according to business logci
    df['member_level'] = df['member_level'].fillna('无')
    df['owner_type'] = df['owner_type'].fillna('个人')
    
    # here filter nan sample for simplicity, actually could be fill up
    all_customer_df = df.dropna()
    print(df.shape)
    print(all_customer_df.shape)
    
    # extract purchase date, here filter nan for simplicity, actually could be fill up
    repair_df = pd.read_csv(os.path.join(path, 'repare_maintain_info1.csv'))
    buy_df = repair_df[['VIN','purchase_date']].drop_duplicates().reset_index(drop=True)
    buy_df = buy_df.dropna()
    
    all_customer_df = all_customer_df.merge(buy_df, on=['VIN'],how='inner')
    
    return all_customer_df

def get_main_feat(path, all_customer_df):
    repair_df = pd.read_csv(os.path.join(path, 'repare_maintain_info1.csv'))
    cols = ['VIN','修理日期','公里数','修理类型']
    repair_df = repair_df[cols]
    repair_df = repair_df[repair_df['VIN'].isin(all_customer_df['VIN'].values)]
    new_cols = ['VIN','date','mile','repair_type']
    repair_df.columns = new_cols
    repair_df = repair_df[new_cols].sort_values(['VIN','date'], ascending=True)
    repair_df = repair_df.dropna()
    repair_df['date'] = pd.to_datetime(repair_df['date']).dt.date
    
    # here filter internal vehicle data
    neibu = repair_df[repair_df['repair_type'].str.contains('内部|二手')][['VIN']].drop_duplicates()
    repair_df_1 = repair_df[~repair_df['VIN'].isin(neibu['VIN'].values)]
    print(repair_df.shape)
    print(repair_df_1.shape)
    
    # here filter non-active in-store maintain sample according to repair_type
    repair_df_2 = repair_df_1[~repair_df_1['repair_type'].str.contains('事故|三包|质量担保|索赔|PDI|返工|免费|返工|售前|代验车|召回|受控')]
    print(repair_df_2.shape)
    print(repair_df_2['repair_type'].unique())
    
    # clean mile data which should be unique about (VIN + date)
    repair_df_3 = repair_df_2.groupby(['VIN','date']).agg(mile = ('mile','mean')).reset_index()
    
    # clean repair_type which should be unique about (VIN + date) and unify their name;
    repair_df_4 = repair_df_2[['VIN','date','repair_type']].drop_duplicates().groupby(['VIN','date'])['repair_type'].agg(lambda x:';'.join(x)).reset_index()
    repair_df_4['repair_type'] = repair_df_4['repair_type'].apply(lambda x:';'.join(natsorted(set(['首次保养' if '首' in i else i for i in x.split(';')]))))
    repair_df_4['repair_type'] = repair_df_4['repair_type'].apply(lambda x:';'.join(natsorted(set(['普通维修' if '普修' in i else i for i in x.split(';')]))))
    
    new_repair_df = repair_df_3.merge(repair_df_4,on=['VIN','date'],how='outer')
    
    return new_repair_df

def get_features(new_repair_df, save_path):
    new_repair_df['date'] = pd.to_datetime(new_repair_df['date'])
    
    data_split_date = new_repair_df['date'].max()+pd.DateOffset(months=-3)
    
    train_x = new_repair_df[new_repair_df['date']<=data_split_date]
    train_y = new_repair_df[new_repair_df['date']>data_split_date]
    
    main_df = train_x.copy()
    main_df = main_df.sort_values(['VIN','date'],ascending=False).reset_index(drop=True)
    
    # extract the recent feat
    feat_1 = main_df.sort_values(['VIN','date'],ascending=False).groupby('VIN').first().reset_index()
    feat_1['last_till_now_days'] = feat_1['date'].apply(lambda x:(pd.to_datetime(today)-pd.to_datetime(x)).days)
    feat_1 = feat_1.rename(columns={'date':'last_date','mile':'last_mile','repair_type':'last_repair_type'})
    
    # shift data to get day/mile diff info
    main_df['relative_last_date'] = main_df.groupby('VIN')['date'].shift(-1)
    main_df['relative_last_mile'] = main_df.groupby('VIN')['mile'].shift(-1)
    main_df = main_df.merge(buy_df[['VIN','purchase_date']].drop_duplicates(), on = 'VIN', how='left')
    main_df['purchase_date'] = pd.to_datetime(main_df['purchase_date'])
    main_df.loc[main_df['relative_last_date'].isna(),'relative_last_date'] = main_df['purchase_date']
    main_df.loc[main_df['relative_last_mile'].isna(),'relative_last_mile'] = 0
    
    del main_df['purchase_date']
    
    main_df['day_diff'] = main_df[['date','relative_last_date']].apply(lambda row:(pd.to_datetime(row[0])-pd.to_datetime(row[1])).days, axis=1, raw=True)
    main_df['mile_diff'] = main_df['mile'] - main_df['relative_last_mile']
    main_df['day_speed'] = main_df['mile_diff']/main_df['day_diff']
    
    # directly filter those data who has wrong purchase date
    main_df_1 = main_df[main_df['day_diff']>0]
    
    # clean mile data, and fill up with statistical method
    fill_df = main_df_1[main_df_1['mile_diff']>=0]
    vin_fill_df = fill_df.groupby('VIN')['day_speed'].median().reset_index().rename(columns={'day_speed':'median_day_speed'})
    main_df_1 = main_df_1.merge(vin_fill_df, on=['VIN'], how='left')
    
    main_df_1.loc[main_df_1['mile_diff']<0, 'relative_last_mile'] = np.nan
    main_df_1.loc[main_df_1['mile_diff']<0, 'day_speed'] = main_df_1['median_day_speed']
    main_df_1.loc[main_df_1['mile_diff']<0, 'mile_diff'] = main_df_1['day_diff']*main_df_1['median_day_speed']
    main_df_1.loc[main_df_1['relative_last_mile'].isna(), 'relative_last_mile'] = main_df_1['mile'] - main_df_1['relative_last_mile']
    
    del main_df_1['median_day_speed']
    
    # filter those users who have left more than three years.
    churn_date = pd.to_datetime(data_split_date)-pd.DateOffset(years=3)
    churn_vin = main_df_1.groupby('VIN')['date'].max().reset_index().rename(columns={'date':'max_date'})
    churn_vin = churn_vin[churn_vin['max_date']<churn_date]
    
    print('all users number: ',main_df_1['VIN'].nunique())
    main_df = main_df_1[~main_df_1['VIN'].isin(churn_vin['VIN'])]
    print('users number after filter loss users: ',main_df['VIN'].nunique())
    
    main_df['rk'] = main_df.sort_values(by=['VIN','date'],ascending=False).groupby('VIN')['date'].rank(method='first')

    # first and second in-store maintain service day and mile diff because almost all users have them;
    feat_2 = main_df[main_df['rk']==1][['VIN','day_diff','mile_diff']].drop_duplicates().rename(columns={'day_diff':'first_to_purchase_day_diff',
                                                                                                         'mile_diff':'first_to_purchase_mile_diff'})
    
    feat_3 = main_df[main_df['rk']==2][['VIN','day_diff','mile_diff']].drop_duplicates().rename(columns={'day_diff':'second_to_first_day_diff',
                                                                                                         'mile_diff':'second_to_first_mile_diff'})
    # extract historical feature
    feat_4 = main_df.groupby('VIN').agg(day_diff_median = ('day_diff','median'), 
                                        day_diff_std = ('day_diff','std'), day_diff_mean = ('day_diff','mean'),
                                        mile_diff_median = ('mile_diff','median'), 
                                        mile_diff_std = ('mile_diff','std'), mile_diff_mean = ('mile_diff','mean'),
                                        day_speed_median = ('day_speed','median'), 
                                        day_speed_std = ('day_speed','std'), day_speed_mean = ('day_speed','mean')
                                       ).reset_index() # 是median，而非medium
    feat_4['day_cv'] = feat_4['day_diff_std'] / feat_4['day_diff_mean']
    feat_4['mile_cv'] = feat_4['mile_diff_std'] / feat_4['mile_diff_mean']
    feat_4['day_speed_cv'] = feat_4['day_speed_std'] / feat_4['day_speed_mean']
    
    for col in ['day_cv', 'mile_cv', 'day_speed_cv']:
        feat_4[col] = feat_4[col].replace([np.inf, -np.inf],np.nan)
        feat_4[col] = feat_4[col].fillna(0)
        feat_4[col] = feat_4[col].astype(float)
    
    for col in ['day_diff_std','mile_diff_std','day_speed_std']:
        feat_4[col] = feat_4[col].fillna(0)
        feat_4[col] = feat_4[col].astype(float)
    
    # further clean data
    nan_vin = feat_4[feat_4['mile_diff_median'].isna()]
    feat_4 = feat_4[~feat_4['VIN'].isin(nan_vin['VIN'])]

    # extract all times for veh maintain service
    feat_5 = main_df.groupby('VIN').agg(all_times = ('date','count')).reset_index()
    
    # extract historical repair type
    feat_6 = main_df.groupby('VIN')['repair_type'].agg(';'.join).rename('all_repair_types').reset_index()
    feat_6['all_repair_types'] = feat_6['all_repair_types'].apply(lambda x:';'.join(natsorted(set(x.split(';')))))
    
    # inner merge them
    feat_df = feat_1.merge(feat_2,on='VIN',how='inner').merge(feat_3,on='VIN',how='inner').merge(feat_4,on='VIN',how='inner').merge(feat_5,on='VIN',how='inner').merge(feat_6,on='VIN',how='inner')
    feat_df = feat_df.merge(all_customer_df, on=['VIN'], how='inner')
    feat_df['car_age'] = feat_df['purchase_date'].apply(lambda x:(pd.to_datetime(today)-pd.to_datetime(x)).days)
    feat_df['car_age'] = feat_df['car_age'].apply(lambda x:math.ceil(x/365))
    
    # estimate next veh maintain date
    feat_df['relative_next_instore_date'] = (
        pd.to_datetime(feat_df['last_date']) + 
        pd.to_timedelta(feat_df['day_diff_median'], unit='D')
    )
    
    feat_df['max_relative_next_instore_date'] = pd.to_datetime(feat_df['relative_next_instore_date']) + pd.DateOffset(months=3)
    
    train_y_df = feat_df[['VIN','last_date','day_diff_median','relative_next_instore_date',
                          'max_relative_next_instore_date']].drop_duplicates().reset_index(drop=True)
    train_y_df['churn_label'] = 0 # still active
    train_y_df.loc[train_y_df['max_relative_next_instore_date'] <= data_split_date, 'churn_label'] = 1  # lost veh customers
    
    feat_df = feat_df.merge(train_y_df[['VIN','churn_label']].drop_duplicates(),on=['VIN'],how='left')
    # print(feat_df.shape)
    # print(feat_df['VIN'].nunique())
    # print(pd.isna(feat_df).sum())
    
    # train and valid data split
    sss =  StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)

    for train_index, val_index in sss.split(feat_df, feat_df['churn_label']):
        train_df = feat_df.iloc[train_index].reset_index(drop=True)
        train_df['dataset'] = 'train'
        val_df = feat_df.iloc[val_index].reset_index(drop=True)
        val_df['dataset'] = 'valid'
    
    all_feat_df = pd.concat([train_df, val_df],axis=0)
    
    all_feat_df.to_csv(os.path.join(save_path,'cleaned_data.csv'),index=False, encoding='utf-8-sig')
    print('cleaned data is saved!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data preprocess')
    
    parser.add_argument('--path', type=str, help='raw data path', default='../cleaned_data')
    parser.add_argument('--save_path', type=str, help='outputs saved path', default='./')
    
    args = parser.parse_args()
    
    all_customer_df = stat_feat(args.path)
    new_repair_df = get_main_feat(args.path, all_customer_df)
    get_features(new_repair_df, args.save_path)  # 修正：移除多余空格


# In[ ]:




