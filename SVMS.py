#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.simplefilter('ignore')
from scipy import stats
from sklearn.datasets.samples_generator import make_blobs
from sklearn.svm import SVC
# faces dataset below from sklearn library.
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
# %%
X,y = make_blobs(n_samples=50,centers=2,cluster_std=0.60,random_state=0)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='summer')
# %%
xfit = np.linspace(-1,3.5)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='summer')
plt.plot([0.6],[2.1],'x',color='green',markeredgewidth=2,markersize=10)
for m,b in [(1,0.65),(0.5,1.6),(-0.2,2.9)]:
    plt.plot(xfit,m*xfit+b,'-k')
plt.xlim(-1,3.5)
# %%
xfit = np.linspace(-1,3.5)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='summer')
for m,b,d in [(1,0.65,0.33),(0.5,1.6,0.55),(-0.2,2.9,0.2)]:
    yfit = m*xfit + b
    plt.plot(xfit,yfit,'-k')
    plt.fill_between(xfit,yfit-d,yfit+d,edgecolor='none',alpha=0.4,color='#AAAAAA')
plt.xlim(-1,3.5)
# %%
model = SVC(kernel='linear',C=1E10)
model.fit(X,y)
# %%
def plot_svc_decision_function(model,ax=None,plot_support=True):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x = np.linspace(xlim[0],xlim[1],30)
    y = np.linspace(ylim[0],ylim[1],30)
    Y,X = np.meshgrid(y,x)
    xy = np.vstack([X.ravel(),Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    ax.contour(X,Y,P,colors='k',levels=[-1,0,1],alpha=0.5,linestyles=['--','-','--'])

    if plot_support:
        ax.scatter(model.support_vectors_[:,0],
                   model.support_vectors_[:,1],s=300,linewidth=1,facecolors='none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)       
# %%
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='summer')
plot_svc_decision_function(model)
# %%
print(model.support_vectors_)
# %%
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)
# %%
fig,ax = plt.subplots(3,5)
for i,axi in enumerate(ax.flat):
    axi.imshow(faces.images[i],cmap='bone')
    axi.set(xticks=[],yticks=[],xlabel=faces.target_names[faces.target[i]])
# %%
pca = RandomizedPCA(n_components=150,whiten=True,random_state=43)
svc = SVC(kernel='rbf',class_weight='balanced')
model = make_pipeline(pca,svc)
# %%
X_train,X_test,y_train,y_test = train_test_split(faces.data,faces.target,random_state=42)
# %%
param_grid = {
    'svc__C':[1,5,10,50],
    'svc__gamma':[0.0001,0.0005,0.001,0.005]
}
grid = GridSearchCV(model,param_grid)
%time grid.fit(X_train,y_train)
print(grid.best_params_)
# %%
model =grid.best_estimator_
yfit = model.predict(X_test)
# %%
fig,ax = plt.subplots(4,6)
for i,axi in enumerate(ax.flat):
    axi.imshow(X_test[i].reshape(62,47),cmap='bone')
    axi.set(xticks=[],yticks=[])
    axi.set_ylabel(faces.target_names[yfit[i]].split()[-1],
                    color='black' if yfit[i] == y_test[i] else 'red')
fig.suptitle('Predicted Names: Incorrect Labels in Red',size=14)
# %%
print(classification_report(y_test,yfit,target_names=faces.target_names))
# %%
cm = confusion_matrix(y_test,yfit)
sns.heatmap(cm.T,square=True,annot=True,fmt='d',cbar=False,xticklabels=faces.target_names,
yticklabels=faces.target_names)
plt.xlabel('True Label')
plt.ylabel('Predicted Label')
# %%
