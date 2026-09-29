from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,ConfusionMatrixDisplay,RocCurveDisplay
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; P=ROOT/'data'/'processed'; M=ROOT/'models'; O=ROOT/'outputs'; A=ROOT/'artifacts'

def main():
    Xtr=np.load(P/'X_train_final.npy'); Xte=np.load(P/'X_test_final.npy'); ytr=np.load(P/'y_train.npy'); yte=np.load(P/'y_test.npy')
    candidates=list(M.glob('*_baseline.pkl'))
    if not candidates: raise FileNotFoundError('No trained model found. Run python src/train.py first.')
    results=[]
    for path in candidates:
        model=joblib.load(path); pred=model.predict(Xte); prob=model.predict_proba(Xte)[:,1]
        results.append({'model':path.stem,'accuracy':accuracy_score(yte,pred),'precision':precision_score(yte,pred),'recall':recall_score(yte,pred),'f1':f1_score(yte,pred),'roc_auc':roc_auc_score(yte,prob)})
    df=pd.DataFrame(results); O.mkdir(exist_ok=True); A.mkdir(exist_ok=True); df.to_csv(O/'model_metrics.csv',index=False)
    best_name=df.sort_values('recall',ascending=False).iloc[0]['model']; model=joblib.load(M/f'{best_name}.pkl'); pred=model.predict(Xte)
    ConfusionMatrixDisplay.from_predictions(yte,pred,display_labels=['No Churn','Churn']); plt.title(f'Confusion Matrix - {best_name}'); plt.tight_layout(); plt.savefig(A/'confusion_matrix.png'); plt.close()
    print(df.to_string(index=False)); print(f'Best model by recall: {best_name}')

if __name__=='__main__': main()
