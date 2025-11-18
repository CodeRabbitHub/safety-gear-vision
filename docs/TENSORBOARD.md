# TensorBoard Integration

TensorBoard visualization is now integrated into the training pipeline for monitoring real-time training metrics.

## Quick Start

### 1. Run Training
```bash
poetry run python scripts/05_train.py
```

Training will automatically log metrics to `logs/tensorboard/{experiment_name}/events.out.tfevents.*`

### 2. Launch TensorBoard
In a new terminal, run:
```bash
poetry run python scripts/09_tensorboard.py
```

### 3. View Metrics
Open your browser and go to: **http://localhost:6006**

## What You Can Monitor

In TensorBoard, you'll see:
- **Scalars**: Training loss, validation loss, accuracy, precision, recall, mAP
- **Training Progress**: Epochs, batch size, learning rate
- **Model Performance**: Class-wise accuracy for all 17 safety gear classes
- **Training Time**: Wall clock time, batches/sec

## Logged Metrics

Default metrics logged by YOLO during training:
- `train/box_loss` - Bounding box regression loss
- `train/cls_loss` - Classification loss
- `train/dfl_loss` - Distribution focal loss
- `val/box_loss` - Validation box loss
- `val/cls_loss` - Validation classification loss
- `metrics/precision` - Precision score
- `metrics/recall` - Recall score
- `metrics/mAP50` - Mean Average Precision @ IoU=0.5
- `metrics/mAP50-95` - Mean Average Precision @ IoU=0.5:0.95

## Accessing Previous Experiments

TensorBoard automatically aggregates all experiments. You can:

1. **Compare multiple runs**: TensorBoard shows all logged experiments in the left sidebar
2. **Filter metrics**: Use the search box to find specific metrics
3. **Switch between experiments**: Click on different experiment names in the left panel

## Directory Structure

Training logs are automatically organized by experiment name:

```
logs/
└── tensorboard/
    ├── exp_20251117_212154/
    │   ├── events.out.tfevents.*
    │   ├── weights/
    │   │   ├── best.pt
    │   │   └── last.pt
    │   └── args.yaml
    ├── exp_20251117_213000/
    │   ├── events.out.tfevents.*
    │   ├── weights/
    │   │   ├── best.pt
    │   │   └── last.pt
    │   └── args.yaml
    └── ...
```

The `setup_project.py` script creates the `logs/` directory. The `logs/tensorboard/` subdirectory and experiment folders are created automatically during training.

## Stopping TensorBoard

Press `Ctrl+C` in the terminal running TensorBoard.

## Troubleshooting

**"No TensorBoard event files found"**
- Make sure training has completed at least one epoch
- Check that `logs/tensorboard/` directory exists
- Run training first: `poetry run python scripts/05_train.py`

**"Port 6006 already in use"**
- TensorBoard is already running in another terminal
- Stop the existing instance first with `Ctrl+C`
- Or use a different port: `tensorboard --logdir logs/tensorboard --port 6007`

## Advanced Usage

### Custom Port
```bash
tensorboard --logdir logs/tensorboard --port 8080
```

### Remote Access
```bash
tensorboard --logdir logs/tensorboard --host 0.0.0.0 --port 6006
```
Then access from another machine: `http://YOUR_IP:6006`

### Logging Custom Metrics
To add custom metrics in training scripts:
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('logs/tensorboard/custom')
writer.add_scalar('custom/metric_name', value, step)
writer.flush()
```

## References

- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)
- [PyTorch TensorBoard Integration](https://pytorch.org/docs/stable/tensorboard.html)
- [Ultralytics YOLOv8 Training](https://docs.ultralytics.com/modes/train/)
