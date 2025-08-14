import seaborn as sns
import matplotlib.pyplot as plt

def churn_distribution(df):
    sns.countplot(x='Churn', data=df)
    plt.title("Churn Distribution")
    plt.show()
