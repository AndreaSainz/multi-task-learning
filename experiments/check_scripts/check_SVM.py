from code.models.SVM import TorchRbfSVM
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import pickle
from torchvision import transforms, datasets


# define SVM sklearn model 
sklearn_model = SVC(kernel="rbf", C=1.0, gamma="scale")

with open("/home/as3628/multi-task-learning/checkpoints/svm/svm_model.pkl", "rb") as f:
    sklearn_model = pickle.load(f)

# extract parameters 
support_vectors = sklearn_model.support_vectors_
dual_coef = sklearn_model.dual_coef_
intercept = sklearn_model.intercept_
gamma = sklearn_model._gamma

# define Pytorch SVM model
pytoech_model = TorchRbfSVM(support_vectors, dual_coef, intercept, gamma)

# DATA
transform = transforms.ToTensor()
test_dataset = datasets.MNIST( root="./data", train=False, download=True, transform=transform )

# for sklearn
X_test_sk = test_dataset.data.view(-1, 28*28).numpy() / 255.0
y_test_sk = test_dataset.targets.numpy()
sk_scores = sklearn_model.decision_function(X_test_sk)

# for pytorch

X_test_torch = torch.tensor(X_test_sk, dtype=torch.float32)
with torch.no_grad():
    torch_scores = pytorch_model(X_test_torch).cpu().numpy()


# differeces
diff = np.abs(torch_scores - sk_scores).mean()
print("Mean absolute difference:", diff)

#predictions
torch_pred = torch_scores.argmax(axis=1)
sk_pred = sklearn_model.predict(X_test_sk)

agreement = (torch_pred == sk_pred).mean()
print("Agreement:", agreement)

