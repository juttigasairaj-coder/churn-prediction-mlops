from pathlib import Path
import joblib, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'data'/'processed'; M=ROOT/'models'; O=ROOT/'outputs'

def main():
    Xtr=np.load(P/'X_train_final.npy'); Xte=np.load(P/'X_test_final.npy'); ytr=np.load(P/'y_train.npy'); yte=np.load(P/'y_test.npy')
    models={'logistic_regression':LogisticRegression(max_iter=1000,random_state=42,class_weight='balanced'),'decision_tree':DecisionTreeClassifier(max_depth=5,random_state=42,class_weight='balanced'),'random_forest':RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42,class_weight='balanced')}
    from sklearn.metrics import recall_score
    best=None; best_recall=-1
    for name,model in models.items():
        model.fit(Xtr,ytr); pred=model.predict(Xte); rec=recall_score(yte,pred)
        joblib.dump(model,M/f'{name}_baseline.pkl'); print(f'{name}: recall={rec:.4f}')
        if rec>best_recall: best_recall=rec; best=(name,model)
    print(f'Best model by recall: {best[0]} ({best_recall:.4f})')

if __name__=='__main__': main()
