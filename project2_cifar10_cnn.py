"""
====================================================================
Project 2: Image Recognition using CNN on CIFAR-10 Dataset
====================================================================
Description:
    This project builds a classifier for image recognition on the
    CIFAR-10 dataset. CIFAR-10 contains 60,000 32x32 color images
    in 10 classes: airplane, automobile, bird, cat, deer, dog, frog,
    horse, ship, and truck.

    Uses scikit-learn's MLPClassifier (Multi-Layer Perceptron) with
    HOG features as the primary approach, compatible with Python 3.14+.

    NOTE: Set USE_TENSORFLOW = True if you have TensorFlow installed.

Libraries Required:
    pip install numpy matplotlib scikit-learn seaborn scikit-image

Author: Acmegrade AI Program
====================================================================
"""

# ========================
# Configuration
# ========================
USE_TENSORFLOW = False  # Set to True if TensorFlow is available

# ========================
# 1. Import Libraries
# ========================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

if USE_TENSORFLOW:
    try:
        from tensorflow import keras
        from tensorflow.keras import layers, models
        from tensorflow.keras.datasets import cifar10
        from tensorflow.keras.utils import to_categorical
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        USING_TF = True
        print("✅ Using TensorFlow/Keras CNN")
    except ImportError:
        USING_TF = False
        print("⚠️  TensorFlow not available. Using scikit-learn approach.")
else:
    USING_TF = False

if not USING_TF:
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    print("🔧 Using scikit-learn classifiers (MLP + Random Forest + SVM)")

# CIFAR-10 Class Names
CLASS_NAMES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']


# ========================
# 2. Load CIFAR-10 Data
# ========================
def load_cifar10_data():
    """
    Load CIFAR-10 dataset. Uses keras.datasets if TF available,
    otherwise downloads via a custom loader.
    """
    print("\n" + "=" * 60)
    print("  Loading CIFAR-10 Dataset")
    print("=" * 60)

    if USING_TF:
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    else:
        # Try loading from keras (even without full TF)
        try:
            from tensorflow.keras.datasets import cifar10
            (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        except ImportError:
            # Download manually using a pickle-based approach
            print("  📥 Downloading CIFAR-10 dataset...")
            import urllib.request
            import tarfile
            import pickle
            import os

            cifar_dir = os.path.join(os.path.expanduser('~'), '.cifar10')
            os.makedirs(cifar_dir, exist_ok=True)

            tar_path = os.path.join(cifar_dir, 'cifar-10-python.tar.gz')

            if not os.path.exists(os.path.join(cifar_dir, 'cifar-10-batches-py')):
                url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
                print(f"  Downloading from {url}...")
                urllib.request.urlretrieve(url, tar_path)
                print("  Extracting...")
                with tarfile.open(tar_path, 'r:gz') as tar:
                    tar.extractall(path=cifar_dir)

            batch_dir = os.path.join(cifar_dir, 'cifar-10-batches-py')

            # Load training data
            X_train_list = []
            y_train_list = []
            for i in range(1, 6):
                batch_file = os.path.join(batch_dir, f'data_batch_{i}')
                with open(batch_file, 'rb') as f:
                    batch = pickle.load(f, encoding='bytes')
                X_train_list.append(batch[b'data'])
                y_train_list.extend(batch[b'labels'])

            X_train = np.concatenate(X_train_list).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            y_train = np.array(y_train_list).reshape(-1, 1)

            # Load test data
            with open(os.path.join(batch_dir, 'test_batch'), 'rb') as f:
                test_batch = pickle.load(f, encoding='bytes')
            X_test = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            y_test = np.array(test_batch[b'labels']).reshape(-1, 1)

    print(f"\n📊 Dataset Statistics:")
    print(f"  Training Images : {X_train.shape}")
    print(f"  Testing Images  : {X_test.shape}")
    print(f"  Image Size      : {X_train.shape[1]}x{X_train.shape[2]}x{X_train.shape[3]} (RGB)")
    print(f"  Classes         : {CLASS_NAMES}")

    return X_train, y_train, X_test, y_test


# ========================
# 3. Visualize Data
# ========================
def visualize_random_grid(X_train, y_train, grid_size=5):
    """Display a random grid of training images."""
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    fig.suptitle('Random CIFAR-10 Training Images', fontsize=16, fontweight='bold')

    for ax in axes.flat:
        idx = np.random.randint(0, len(X_train))
        ax.imshow(X_train[idx])
        label = y_train[idx][0] if len(y_train[idx].shape) > 0 else y_train[idx]
        ax.set_title(CLASS_NAMES[label], fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('cifar10_random_grid.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  ✅ Random grid saved as 'cifar10_random_grid.png'")


def plot_class_distribution(y_train, y_test):
    """Plot the distribution of classes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('CIFAR-10 Class Distribution', fontsize=14, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    unique, counts = np.unique(y_train, return_counts=True)
    ax1.bar(range(10), counts, color=colors, edgecolor='black')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    ax1.set_title('Training Set')
    ax1.set_xticks(range(10))
    ax1.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')

    unique, counts = np.unique(y_test, return_counts=True)
    ax2.bar(range(10), counts, color=colors, edgecolor='black')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Count')
    ax2.set_title('Test Set')
    ax2.set_xticks(range(10))
    ax2.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('cifar10_class_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  ✅ Distribution saved as 'cifar10_class_distribution.png'")


# ========================
# 4. Feature Extraction (for scikit-learn)
# ========================
def extract_features(X_train, X_test):
    """
    Extract features from images for scikit-learn classifiers.
    Uses color histograms + pixel features.
    """
    print("\n⚙️  Extracting Features from Images...")

    def image_to_features(images):
        """Convert images to feature vectors."""
        n = len(images)
        features_list = []

        for i in range(n):
            img = images[i].astype('float32') / 255.0
            feat = []

            # 1. Flattened pixel values (downsampled to 16x16)
            from PIL import Image as PILImage
            try:
                pil_img = PILImage.fromarray(images[i])
                small = np.array(pil_img.resize((16, 16))).flatten() / 255.0
                feat.extend(small)
            except ImportError:
                # Simple downsampling without PIL
                small = img[::2, ::2, :].flatten()
                feat.extend(small)

            # 2. Color histogram features (per channel)
            for c in range(3):
                hist, _ = np.histogram(img[:, :, c], bins=32, range=(0, 1))
                hist = hist / hist.sum()  # Normalize
                feat.extend(hist)

            # 3. Mean and std per channel
            for c in range(3):
                feat.append(img[:, :, c].mean())
                feat.append(img[:, :, c].std())

            # 4. Grayscale statistics
            gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
            feat.append(gray.mean())
            feat.append(gray.std())

            features_list.append(feat)

        return np.array(features_list)

    print("  Processing training images...")
    X_train_feat = image_to_features(X_train)
    print(f"  ✅ Training features: {X_train_feat.shape}")

    print("  Processing test images...")
    X_test_feat = image_to_features(X_test)
    print(f"  ✅ Test features: {X_test_feat.shape}")

    # Scale features
    scaler = StandardScaler()
    X_train_feat = scaler.fit_transform(X_train_feat)
    X_test_feat = scaler.transform(X_test_feat)
    print("  ✅ Features scaled")

    return X_train_feat, X_test_feat, scaler


# ========================
# 5. Build and Train Models
# ========================
def train_sklearn_models(X_train, y_train, X_test, y_test):
    """Train multiple scikit-learn classifiers and compare."""
    print("\n🤖 Training Multiple Classifiers...")
    print("=" * 60)

    y_train_flat = y_train.flatten()
    y_test_flat = y_test.flatten()

    classifiers = {
        'MLP Neural Network': MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=256,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=50,
            random_state=42,
            verbose=True,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=30,
            random_state=42,
            n_jobs=-1,
            verbose=1
        ),
    }

    results = {}

    for name, clf in classifiers.items():
        print(f"\n  📌 Training {name}...")
        print("-" * 40)

        clf.fit(X_train, y_train_flat)

        # Predict
        y_pred = clf.predict(X_test)

        # Metrics
        accuracy = (y_pred == y_test_flat).mean()

        results[name] = {
            'model': clf,
            'accuracy': accuracy,
            'y_pred': y_pred
        }

        print(f"\n  ✅ {name} Test Accuracy: {accuracy * 100:.2f}%")

    return results, y_test_flat


def train_tensorflow_model(X_train, y_train, X_test, y_test):
    """Train a CNN model with TensorFlow."""
    print("\n🏗️  Building CNN Model for CIFAR-10...")

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    y_train_enc = to_categorical(y_train, 10)
    y_test_enc = to_categorical(y_test, 10)

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1,
                                  height_shift_range=0.1, horizontal_flip=True)
    datagen.fit(X_train)

    history = model.fit(datagen.flow(X_train, y_train_enc, batch_size=64),
                        epochs=50, validation_data=(X_test, y_test_enc),
                        callbacks=[keras.callbacks.EarlyStopping(
                            monitor='val_loss', patience=5, restore_best_weights=True)],
                        steps_per_epoch=len(X_train) // 64, verbose=1)

    return model, history, X_test, y_test_enc


# ========================
# 6. Compare and Visualize
# ========================
def compare_models(results):
    """Compare model accuracies."""
    print("\n📊 Model Comparison:")
    print("=" * 60)

    names = list(results.keys())
    accuracies = [results[n]['accuracy'] * 100 for n in names]

    for name in names:
        acc = results[name]['accuracy'] * 100
        bar = "█" * int(acc / 2)
        print(f"  {name:<25s}: {acc:.2f}% {bar}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['steelblue', 'coral', 'mediumseagreen']
    bars = ax.bar(names, accuracies, color=colors[:len(names)], edgecolor='black')

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('CIFAR-10 Model Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, max(accuracies) + 10])

    plt.tight_layout()
    plt.savefig('cifar10_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  ✅ Comparison chart saved as 'cifar10_model_comparison.png'")

    best_name = max(results, key=lambda x: results[x]['accuracy'])
    print(f"\n  🏆 Best Model: {best_name} ({results[best_name]['accuracy']*100:.2f}%)")
    return best_name


# ========================
# 7. Detailed Analysis
# ========================
def detailed_analysis(y_true, y_pred, model_name):
    """Show detailed metrics for the best model."""
    print(f"\n📋 Detailed Analysis: {model_name}")
    print("=" * 60)

    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, linecolor='gray')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('cifar10_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  ✅ Confusion matrix saved as 'cifar10_confusion_matrix.png'")


# ========================
# 8. Per-Class Accuracy
# ========================
def per_class_accuracy(y_true, y_pred):
    """Show accuracy for each class."""
    print("\n📊 Per-Class Accuracy:")
    print("-" * 45)

    cm = confusion_matrix(y_true, y_pred)
    class_acc = cm.diagonal() / cm.sum(axis=1)

    for i, name in enumerate(CLASS_NAMES):
        bar = "█" * int(class_acc[i] * 30)
        print(f"  {name:12s} : {class_acc[i]*100:6.2f}% {bar}")

    print(f"\n  {'Average':12s} : {np.mean(class_acc)*100:6.2f}%")


# ========================
# 9. Visualize Predictions
# ========================
def visualize_predictions(model, X_test, y_test, X_test_images=None, num_samples=15, is_tf=False):
    """Show predictions on random test images."""
    print("\n🔮 Predictions on Random Test Images:")

    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    fig.suptitle('CIFAR-10 Predictions', fontsize=14, fontweight='bold')

    indices = np.random.choice(len(X_test), num_samples, replace=False)

    for i, ax in enumerate(axes.flat):
        idx = indices[i]

        if is_tf:
            ax.imshow(X_test[idx])
            pred = model.predict(X_test[idx:idx+1], verbose=0)
            pred_class = CLASS_NAMES[np.argmax(pred)]
            true_class = CLASS_NAMES[np.argmax(y_test[idx])]
            conf = np.max(pred) * 100
        else:
            if X_test_images is not None:
                ax.imshow(X_test_images[idx])
            else:
                ax.imshow(np.zeros((32, 32, 3)))

            pred = model.predict(X_test[idx:idx+1])
            pred_class = CLASS_NAMES[pred[0]]
            true_label = y_test[idx] if isinstance(y_test[idx], (int, np.integer)) else y_test[idx]
            true_class = CLASS_NAMES[true_label]

            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test[idx:idx+1])
                conf = np.max(proba) * 100
            else:
                conf = 0

        color = 'green' if pred_class == true_class else 'red'
        ax.set_title(f'Pred: {pred_class}\nTrue: {true_class}\n'
                     f'Conf: {conf:.1f}%', fontsize=9, color=color, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('cifar10_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  ✅ Predictions saved as 'cifar10_predictions.png'")


# ========================
# MAIN EXECUTION
# ========================
if __name__ == "__main__":
    print("╔" + "═" * 58 + "╗")
    print("║  Image Recognition on CIFAR-10 Dataset                  ║")
    if USING_TF:
        print("║  Method: Convolutional Neural Network (TensorFlow)      ║")
    else:
        print("║  Method: MLP + Random Forest (scikit-learn)             ║")
    print("╚" + "═" * 58 + "╝")

    # Step 1: Load Data
    X_train, y_train, X_test, y_test = load_cifar10_data()

    # Step 2: Visualize
    visualize_random_grid(X_train, y_train)
    plot_class_distribution(y_train, y_test)

    if USING_TF:
        # TensorFlow path
        model, history, X_test_norm, y_test_enc = train_tensorflow_model(
            X_train, y_train, X_test, y_test)

        y_pred = np.argmax(model.predict(X_test_norm, verbose=0), axis=1)
        y_true = y_test.flatten()

        detailed_analysis(y_true, y_pred, 'CNN (TensorFlow)')
        per_class_accuracy(y_true, y_pred)
        visualize_predictions(model, X_test_norm, y_test_enc,
                              X_test_images=X_test, is_tf=True)
        model.save('cifar10_cnn_model.h5')
    else:
        # Scikit-learn path
        # Step 3: Extract Features
        X_train_feat, X_test_feat, scaler = extract_features(X_train, X_test)

        # Step 4: Train Models
        results, y_test_flat = train_sklearn_models(
            X_train_feat, y_train, X_test_feat, y_test)

        # Step 5: Compare
        best_name = compare_models(results)

        # Step 6: Detailed Analysis
        best_model = results[best_name]
        detailed_analysis(y_test_flat, best_model['y_pred'], best_name)

        # Step 7: Per-Class Accuracy
        per_class_accuracy(y_test_flat, best_model['y_pred'])

        # Step 8: Visualize Predictions
        visualize_predictions(best_model['model'], X_test_feat, y_test_flat,
                              X_test_images=X_test, is_tf=False)

    print("\n" + "=" * 60)
    print("✅ Project 2 Complete: CIFAR-10 Image Recognition")
    print("=" * 60)
