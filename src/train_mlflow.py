import os, json, joblib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay
ROOT=Path(__file__).resolve().parents[1]; P=ROOT/'data'/'processed'; M=ROOT/'models'; A=ROOT/'artifacts'

def train_and_track(run_name='RandomForest', params=None):
    if params is None: params={'n_estimators':100,'max_depth':10,'random_state':42,'class_weight':'balanced'}
    X_train=np.load(P/'X_train_final.npy'); X_test=np.load(P/'X_test_final.npy'); y_train=np.load(P/'y_train.npy'); y_test=np.load(P/'y_test.npy')
    mlflow.set_experiment('Telco_Churn_Prediction')
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params); mlflow.log_param('model_family','RandomForest')
        model=RandomForestClassifier(**params); model.fit(X_train,y_train)
        y_pred=model.predict(X_test); y_prob=model.predict_proba(X_test)[:,1]
        metrics={'accuracy':accuracy_score(y_test,y_pred),'precision':precision_score(y_test,y_pred),'recall':recall_score(y_test,y_pred),'f1_score':f1_score(y_test,y_pred),'roc_auc':roc_auc_score(y_test,y_prob)}
        mlflow.log_metrics(metrics); print(f"Metrics logged: F1 = {metrics['f1_score']:.4f} | ROC-AUC = {metrics['roc_auc']:.4f}")
        A.mkdir(exist_ok=True)
        cm=A/'confusion_matrix_mlflow.png'; fig,ax=plt.subplots(figsize=(6,5)); ConfusionMatrixDisplay.from_predictions(y_test,y_pred,ax=ax,cmap='Blues'); fig.savefig(cm,bbox_inches='tight'); plt.close(fig); mlflow.log_artifact(str(cm),artifact_path='plots')
        roc=A/'roc_curve_mlflow.png'; fig,ax=plt.subplots(figsize=(6,5)); RocCurveDisplay.from_predictions(y_test,y_prob,ax=ax); fig.savefig(roc,bbox_inches='tight'); plt.close(fig); mlflow.log_artifact(str(roc),artifact_path='plots')
        meta=P/'dataset_metadata.json'
        if meta.exists(): mlflow.log_artifact(str(meta),artifact_path='metadata')
        mlflow.sklearn.log_model(sk_model=model,name='model')
        M.mkdir(exist_ok=True); joblib.dump(model,M/'random_forest_model.pkl')
        print(f"Run '{run_name}' successfully tracked!")

if __name__=='__main__': train_and_track()
