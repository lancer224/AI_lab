import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

    def logsig(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    def logsig_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.logsig(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.logsig(self.z2)
        return self.a2

    def backward(self, X, y, output, learning_rate):
        m = X.shape[0]

        dZ2 = output - y
        dW2 = (1 / m) * np.dot(self.a1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.logsig_derivative(self.a1)
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, epochs, learning_rate, X_val=None, y_val=None):
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(epochs):
            output = self.forward(X)

            # Обратное распространение
            self.backward(X, y, output, learning_rate)

            # Сохранение метрик каждые 100 эпох
            if epoch % 100 == 0:
                train_loss = np.mean(np.square(y - output))
                train_losses.append(train_loss)

                # Точность на обучающих данных
                train_pred = (output > 0.5).astype(int)
                train_acc = np.mean(np.all(train_pred == y, axis=1))
                train_accuracies.append(train_acc)

                # Валидация
                if X_val is not None and y_val is not None:
                    val_output = self.forward(X_val)
                    val_loss = np.mean(np.square(y_val - val_output))
                    val_losses.append(val_loss)

                    val_pred = (val_output > 0.5).astype(int)
                    val_acc = np.mean(np.all(val_pred == y_val, axis=1))
                    val_accuracies.append(val_acc)

            # Вывод прогресса каждые 1000 эпох
            if epoch % 1000 == 0:
                if X_val is not None and y_val is not None:
                    print(f"Шаг {epoch}, Ошибка: {train_loss:.6f}, Точность: {train_acc:.4f}, "
                          f"Валидационная ошибка: {val_loss:.6f}, Валидационная точность: {val_acc:.4f}")
                else:
                    print(f"Шаг {epoch}, Ошибка: {train_loss:.6f}, Точность: {train_acc:.4f}")

        return train_losses, val_losses, train_accuracies, val_accuracies

    def predict(self, X, threshold=0.5):
        output = self.forward(X)
        return (output > threshold).astype(int)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(np.all(predictions == y, axis=1))
        return accuracy, predictions


def load_data(input_file, output_file=None):
    try:
        X = np.loadtxt(input_file)
        if X.shape[1] != 12:
            if X.shape[1] > 12:
                X = X[:, :12]

        if output_file:
            y = np.loadtxt(output_file)

            if y.ndim == 1:
                if len(y) % 2 == 0:
                    y = y.reshape(-1, 2)

            if y.shape[1] != 2:
                if y.shape[1] > 2:
                    y = y[:, :2]
        return X, y

    except Exception as e:
        return None, None


def save_predictions(predictions, output_file):
    np.savetxt(output_file, predictions, fmt='%d', delimiter=' ')


def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):

    epochs = range(0, len(train_losses) * 100, 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # График ошибки
    ax1.plot(epochs, train_losses, 'b-', label='Ошибка обучения', linewidth=2)
    if val_losses:
        ax1.plot(epochs, val_losses, 'r-', label='Ошибка валидации', linewidth=2)
    ax1.set_title('История обучения - Ошибка')
    ax1.set_xlabel('Шаг')
    ax1.set_ylabel('Ошибка')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # График точности
    ax2.plot(epochs, train_accuracies, 'b-', label='Точность обучения', linewidth=2)
    if val_accuracies:
        ax2.plot(epochs, val_accuracies, 'r-', label='Точность валидации', linewidth=2)
    ax2.set_title('История обучения - Точность')
    ax2.set_xlabel('Шаг')
    ax2.set_ylabel('Точность')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    input_size = 12
    hidden_size = 8
    output_size = 2
    learning_rate = 0.1
    epochs = 5000

    nn = NeuralNetwork(input_size, hidden_size, output_size)

    X_train, y_train = load_data(
        'C:/Users/gobli/Documents/GitHub/AI_lab/project_folder/files/dataIn.txt',
        'C:/Users/gobli/Documents/GitHub/AI_lab/project_folder/files/dataOut.txt'
    )

    if X_train is None or y_train is None:
        np.random.seed(42)
        X_train = np.random.randint(0, 2, (1000, 12))
        y_train = np.column_stack([
            (X_train[:, 0] & X_train[:, 1] | X_train[:, 2]).astype(int),
            (X_train[:, 3] ^ X_train[:, 4]).astype(int)
        ])

    split_idx = int(0.8 * len(X_train))
    X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
    y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

    train_losses, val_losses, train_accuracies, val_accuracies = nn.train(
        X_tr, y_tr, epochs, learning_rate, X_val, y_val
    )

    train_accuracy, _ = nn.evaluate(X_train, y_train)
    print(f"Точность на обучающих данных: {train_accuracy:.4f}")

    X_test, y_test = load_data(
        'C:/Users/gobli/Documents/GitHub/AI_lab/project_folder/files/test_input.txt',
        'C:/Users/gobli/Documents/GitHub/AI_lab/project_folder/files/test_output.txt'
    )


    test_accuracy, test_predictions = nn.evaluate(X_test, y_test)
    print(f"Точность на тестовых данных: {test_accuracy:.4f}")

    save_predictions(test_predictions, 'model_predictions.txt')
    print("\nПримеры предсказаний (первые 5):")
    for i in range(min(5, len(X_test))):
        print(f"Вход: {X_test[i]} -> Истино: {y_test[i]} -> Предсказано: {test_predictions[i]}")


    # 5. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
    print("\n5. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    print("-" * 30)
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)

    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТЫ:")
    print(f"Точность на обучении: {train_accuracy:.4f}")
    if test_accuracy > 0:
        print(f"Точность на тесте: {test_accuracy:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()