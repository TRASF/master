import os
import yaml
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from src.framework.supervised.dataset import SupervisedDataset

class ModelEvaluator:
    """
    Handles model evaluation on a test dataset.
    """
    def __init__(self, config_path: str = "configs/defaults.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.config_path = config_path
        self.dataset_manager = SupervisedDataset(config_path=config_path)
        self.class_dict = self.config['class_dict']
        self.all_class_names = [k for k, v in sorted(self.class_dict.items(), key=lambda item: item[1])]

    def evaluate(self, model_path: str, dataset_path: str, output_dir: str = "evaluation_results"):
        """
        Loads the model and dataset from path, runs evaluation, and saves results.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 1. Load Model
        print(f"Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path)

        # 2. Load Test Dataset
        batch_size = self.config['training']['batch_size']
        print(f"Loading test dataset from {dataset_path}...")
        
        test_ds = self.dataset_manager.load_for_training(
            dataset_path, 
            split='test', 
            batch_size=batch_size, 
            shuffle=False
        )
        
        return self.evaluate_dataset(model, test_ds, output_dir)

    def evaluate_dataset(self, model, dataset, output_dir: str = "evaluation_results"):
        """
        Runs evaluation on a provided tf.data.Dataset object.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 1. Predict
        print("Running predictions...")
        y_true = []
        y_pred_probs = []

        for x, y in dataset:
            preds = model.predict(x, verbose=0)
            y_true.extend(y.numpy())
            y_pred_probs.extend(preds)

        y_true = np.array(y_true)
        y_pred_probs = np.array(y_pred_probs)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # 2. Compute Metrics
        print("Computing metrics...")
        
        # Filter target names to match classes present in y_true and y_pred
        present_labels = sorted(list(set(y_true) | set(y_pred)))
        present_class_names = [self.all_class_names[i] for i in present_labels]

        report = classification_report(y_true, y_pred, labels=present_labels, target_names=present_class_names, output_dict=True)
        report_text = classification_report(y_true, y_pred, labels=present_labels, target_names=present_class_names)
        
        print("\nClassification Report:")
        print(report_text)

        # Save report to text file
        with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
            f.write(report_text)

        # 3. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=present_labels)
        self.plot_confusion_matrix(cm, present_class_names, output_dir)
        self.save_confusion_matrix_text(cm, present_class_names, output_dir)

        # 4. Accuracy
        accuracy = np.mean(y_true == y_pred)
        print(f"\nOverall Test Accuracy: {accuracy:.4f}")
        
        with open(os.path.join(output_dir, "metrics.yaml"), "w") as f:
            yaml.dump({"test_accuracy": float(accuracy)}, f)

        print(f"Evaluation results saved to {output_dir}")
        return report

    def save_confusion_matrix_text(self, cm, class_names, output_dir):
        """
        Saves the confusion matrix as a formatted text file.
        """
        output_path = os.path.join(output_dir, "confusion_matrix.txt")
        header = r"True \ Predicted | " + " | ".join(class_names)
        separator = "-" * len(header)
        
        with open(output_path, "w") as f:
            f.write("Confusion Matrix (Rows: True, Cols: Predicted)\n")
            f.write(separator + "\n")
            f.write(header + "\n")
            f.write(separator + "\n")
            
            for i, row in enumerate(cm):
                row_str = f"{class_names[i]:<16} | " + " | ".join([f"{count:^10}" for count in row])
                f.write(row_str + "\n")
            f.write(separator + "\n")
        
        print(f"Confusion matrix text saved to {output_path}")

    def plot_confusion_matrix(self, cm, class_names, output_dir):
        """
        Plots and saves the confusion matrix.
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap='Blues', xticks_rotation='vertical', ax=ax)
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a trained MosSongPlus model.")
    parser.add_argument("--model", type=str, default="models/supervised_mossongplus/final_model.keras", help="Path to the trained model.")
    parser.add_argument("--dataset", type=str, default="dataset/tf_supervised_test", help="Path to the TF dataset directory.")
    parser.add_argument("--output", type=str, default="evaluation_results", help="Directory to save evaluation results.")
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator()
    evaluator.evaluate(args.model, args.dataset, args.output)
