#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import os, dotenv, gc, argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
import chromadb
from langchain_core.documents import Document
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

def auto_scale(df, num_cols):
    df_scaled = df.copy()
    for col in num_cols:
        col_data = df[col].values.reshape(-1,1)
        q1, q3 = np.percentile(col_data, [25,75])
        iqr = q3 - q1
        skew = pd.Series(col_data.flatten()).skew()
        if iqr > 0 and (np.max(col_data) - np.min(col_data)) / iqr > 10:
            scaler = RobustScaler()
        elif abs(skew) > 1:
            scaler = PowerTransformer(method='yeo-johnson')
        else:
            scaler = StandardScaler()
        df_scaled[col] = scaler.fit_transform(col_data)
    return df_scaled

def llm_data_preprocess(save_path):
    dff = pd.read_csv(os.path.join(save_path,'cleaned_data.csv'))
    
    num_cols = ['last_mile', 'last_till_now_days', 'first_to_purchase_day_diff', 'first_to_purchase_mile_diff', 'second_to_first_day_diff',
                'second_to_first_mile_diff', 'day_diff_median', 'mile_diff_median',
                'day_speed_median', 'day_cv', 'mile_cv', 'day_speed_cv',
                'all_times','car_age']
    text_cols = ['last_repair_type', 'all_repair_types','owner_type','car_mode','car_level','member_level']
    
    dff = dff[num_cols + text_cols]
    print(dff.shape)
    print(dff['VIN'].nunique())
    
    # for below cols could be one-hot-features as well as textual features, here I adapted as new try on textual features which will be feed into LLM.
    dff['member_level'] = dff['member_level'].apply(lambda x:'会员卡：'+x)
    dff['owner_type'] = dff['owner_type'].apply(lambda x:'用户性质：'+x)
    dff['car_mode'] = dff['car_mode'].apply(lambda x:'汽车型号：'+x)
    
    def car_level_bin(x):
        if x=='family_1':
            return '高档车'
        elif x == 'family_2':
            return '中档车'
        else:
            return '低档车'
    dff['car_level'] = dff['car_level'].apply(lambda x:car_level_bin(x))
    
    # these two features could be removed from LLM embedding model feeding for OpenAI token cost;
    dff['last_repair_type'] = dff['last_repair_type'].apply(lambda x:'上次进店类型：'+x)
    dff['all_repair_types'] = dff['all_repair_types'].apply(lambda x:'历史进店类型：'+x)
    
    # automatically normalization according to numerical feature's character
    dff = auto_scale(dff, num_cols)
    
    # map label in chinese
    label_map = {1:'用户标签：流失', 0:'用户标签：未流失'}
    dff['label_txt'] = dff['churn_label'].map(label_map)
    
    # key used in KNN-strategy inference
    dff['key_label'] = dff[['VIN','churn_label']].apply(lambda row:row[0]+':'+str(row[1]), axis=1, raw=True)
    
    return dff

def train(dff, save_path):
    train = dff[dff['dataset']=='train']
    valid = dff[dff['dataset']=='valid']
    train_idx = len(train)-1
    dt = pd.concat([train,valid], axis=0)
    
    dt['text_feature'] = dt[['last_repair_type','all_repair_types','owner_type', 'car_mode', 'car_level',
                             'member_level']].apply(lambda row:row[0]+'，'+row[1]+'，'+row[2]+'，'+row[3]+'，'+row[4]+'，'+row[5], raw=True, axis=1)
    
    texts = dt['text_feature'].values.tolist()
    key_ids = dt['key_label'].values.tolist()
    
    # create docs text for llm embeddding model
    docs = [
        Document(
            page_content=text,
            metadata={"key_id": key_id}
        ) 
        for text, key_id in zip(texts, key_ids)
    ]
    print(f"doc number: {len(docs)}")
    
    # Embedding model which is important after testing, if change SentenceTransformer, the representative performance will decreas.
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )
    
    # even it is 1536 dim, not recommend reduce dim by PCA method, the model performance will be degraded greatly
    text_embedding = embeddings_model.embed_documents(texts)
    text_embedding = np.array(text_embedding)
    print(f"textual embedding shape: {text_embedding.shape}")
    
    num_embedding = dt[num_cols].to_numpy()
    print(f"number embedding shape: {num_embedding.shape}")
    
    weight_embedding = np.hstack([text_embedding * 0.3, num_embedding * 0.7])
    print(f"weighted embedding shape: {weight_embedding.shape}")
    
    final_embedding = normalize(weight_embedding)
    print(f"final embedding shape: {final_embedding.shape}")
    
    train_embedding = final_embedding[:train_idx]
    valid_embedding = final_embedding[train_idx:]
    print(f"train datasets embedding shpae: {train_embedding.shape}")
    print(f"valid datasets embedding shape: {valid_embedding.shape}")
    
    train_docs = docs[:train_idx]
    valid_docs = docs[train_idx:]
    print(f"train docs number (text feature): {len(train_docs)}")
    print(f"valid docs number (text feature): {len(valid_docs)}")
    
    #############create chromadb client to save above custom final embedding#######################
    persist_directory = "./chroma_db"
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    
    # # update the chromadb
    # try:
    #     chroma_client.delete_collection(name="multimodal")
    # except:
    #     pass
    
    collection = chroma_client.create_collection(name="multimodal")

    train_texts = [doc.page_content for doc in train_docs]
    train_metadatas = [doc.metadata for doc in train_docs]
    train_ids = [f"doc_{i}" for i in range(len(train_docs))]
    train_embeddings = train_embedding.tolist()
    
    batch_size = 5000
    total_batches = (len(train_docs) + batch_size - 1) // batch_size
    
    print(f"begin add embedding data by batch，total docs number: {len(train_docs)}，batch size: {batch_size}，batch number: {total_batches}")
    for i in range(0, len(train_docs), batch_size):
        batch_end = min(i + batch_size, len(train_docs))
        batch_num = i // batch_size + 1
        
        print(f"Dealing with {batch_num}/{total_batches} (Doc {i} to {batch_end})")
        
        collection.add(
            embeddings=train_embeddings[i:batch_end],
            documents=train_texts[i:batch_end],
            metadatas=train_metadatas[i:batch_end],
            ids=train_ids[i:batch_end]
        )
        
    print("All embedding saving is completed!")
    
    # create LangChain vector store in order to do RAG for KNN-strategy inference
    vectorstore = Chroma(
        client=chroma_client,
        collection_name="multimodal",
        embedding_function=None
    )

    # start to predict one by one
    res = []
    for i in range(len(valid_docs)):
        query_vec = valid_embedding[i].tolist()
        query_doc = valid_docs[i]
        
        results_with_scores = vectorstore.similarity_search_by_vector_with_relevance_scores(
            embedding=query_vec,
            k=10
        )
        
        knn_cal=0
        for i, (doc, score) in enumerate(results_with_scores):
            knn_cal+=int(doc.metadata['key_id'].split(':')[1])
        res.append([query_doc.metadata['key_id'], 1 if (knn_cal/10)>=0.40 else 0, knn_cal])
    
    res_df = pd.DataFrame(data=res, columns=['key_label','pred_label','total_score'])
    res_df['VIN'] = res_df['key_label'].apply(lambda x:x.split(':')[0])
    res_df['true_label'] = res_df['key_label'].apply(lambda x:x.split(':')[1])
    res_df.to_csv(os.path.join(save_path, 'prediction.csv'), index=False, encoding='utf-8-sig')

def metric(save_path):
    res_df = pd.read_csv(os.path.join(save_path, 'prediction.csv'))
    for col in ['true_label','pred_label']:
        res_df[col] = res_df[col].astype(float)
        res_df[col] = res_df[col].astype(int)
    
    y_true = res_df['true_label'].values
    y_pred = res_df['pred_label'].values
    auc_score = roc_auc_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"AUC Score: {auc_score:.3f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model predict')
    parser.add_argument('--save_path', type=str, help='path to save results', default='./')
    args = parser.parse_args()
    
    dff = llm_data_preprocess(args.save_path)
    train(dff)
    metrics(args.save_path)

