import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm
from DatasetLoader import train_dataloader_simple, test_dataloader_simple
from loop_train_test import train


# 1 ========================= Builting Model CNN ==================================
class fire_Classifier_CNN(nn.Module):
    def __init__(self, input_shape:int, hidden_shape:int, output_shape:int):
        super().__init__()
        # block one have 3 conv2d with maxpooling 2 * 2
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_shape,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_shape,
                      out_channels=hidden_shape,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_shape,
                      out_channels=hidden_shape,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # block one have 2 conv2d with maxpooling 2 * 2
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_shape,
                      out_channels=hidden_shape,
                      kernel_size=3,
                      stride=1,
                      padding=1),
                      nn.ReLU(),
            nn.Conv2d(in_channels=hidden_shape,
                      out_channels=hidden_shape,
                      kernel_size=3,
                      stride=1,
                      padding=1),
                      nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
                  nn.Flatten(),
            # Where did this in_features shape come from?
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            # 64 to 2*2 maxpooling = 32 ->  32 to maxpooling 2*2 =16 
                  nn.Linear(in_features=hidden_shape * 16 * 16 , out_features=output_shape)
              )
        
    def forward(self, x: torch.Tensor):
           x = self.block_1(x)
           x = self.block_2(x)
          #  print(x.shape)
           print(x.shape)
           x = self.classifier(x)
           return x


# 2 ========================= device Set random seeds ==================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
model = fire_Classifier_CNN(input_shape=3,
                  hidden_shape=10,
                  output_shape=2).to(device)

print(model)
image_batch, label_batch = next(iter(train_dataloader_simple))
print(image_batch.shape), print(label_batch.shape)


model(image_batch.to(device))

torch.cuda.manual_seed(42)


# 3 ========================= Training and Testing Model ==================================
NUM_EPOCHS = 12 # Set number of epochs

loss_fn = nn.CrossEntropyLoss() # Setup loss function and optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)


from timeit import default_timer as timer  # Start the timer
start_time = timer()

# Train model
model_result = train(model=model,
                        train_dataloader=train_dataloader_simple,
                        test_dataloader=test_dataloader_simple,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=NUM_EPOCHS)

# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")


# acc dataFreame
model_dataframs_acc = pd.DataFrame(model_result)
print(model_dataframs_acc)

# 4 =============================  Save model weights  ===========================
torch.save(model.state_dict(), "fire_cnn_model.pth")
print("Model weights saved successfully!")
print(device)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# ===============================  Done ============================