from ultralytics import YOLO
from itertools import product

def main():
    model = YOLO("best.pt")
    train_results = model.train(data='data.yaml',
        epochs=500,
        imgsz=640,
        device=0)

    # evaluate performace on the validation set
    metrics = model.val()
    # perform obj detection on image
    results = model("datasets/test/img8632.png")
    results[0].show()
    # export format
    export_model = model.export(format="onnx")

def hyperparameter_tuning():
    # Define hyperparameter ranges to explore
    epochs_options = [100, 200, 500]
    imgsz_options = [640, 960]
    lr0_options = [0.001, 0.01, 0.02]
    momentum_options = [0.8, 0.9, 0.95]
    weight_decay_options = [0.0005, 0.001, 0.005]

    best_metrics = None
    best_params = None

    # Iterate over all combinations of hyperparameters
    for epochs, imgsz, lr0, momentum, weight_decay in product(
        epochs_options, imgsz_options, lr0_options, momentum_options, weight_decay_options):
        
        print(f"Training with epochs={epochs}, imgsz={imgsz}, lr0={lr0}, momentum={momentum}, weight_decay={weight_decay}")

        # Initialize model
        model = YOLO("best.pt")
        
        # Train model with current hyperparameters
        train_results = model.train(
            data='data.yaml',
            epochs=epochs,
            imgsz=imgsz,
            device=0,
            lr0=lr0,
            momentum=momentum,
            weight_decay=weight_decay
        )
        
        # Evaluate on validation set
        metrics = model.val()
        
        # Track best metrics and parameters
        if best_metrics is None or metrics['mAP_0.5'] > best_metrics['mAP_0.5']:
            best_metrics = metrics
            best_params = {
                "epochs": epochs,
                "imgsz": imgsz,
                "lr0": lr0,
                "momentum": momentum,
                "weight_decay": weight_decay
            }

    print("Best hyperparameters found:")
    print(best_params)
    print("With metrics:")
    print(best_metrics)

    # Optional: train final model with best params
    final_model = YOLO("best.pt")
    final_model.train(
        data='data.yaml',
        epochs=best_params["epochs"],
        imgsz=best_params["imgsz"],
        device=0,
        lr0=best_params["lr0"],
        momentum=best_params["momentum"],
        weight_decay=best_params["weight_decay"]
    )

    # Evaluate and export the final model
    final_metrics = final_model.val()
    results = final_model("datasets/test/img8632.png")
    results[0].show()
    final_model.export(format="onnx")

if __name__ == "__main__":
    hyperparameter_tuning()