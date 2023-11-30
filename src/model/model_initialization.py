
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Assuming variables like CNN_LAYERS, CLASS, EMB_SIZE, nHEADS, D_HID, and nLAYERS are defined elsewhere in your code
modelCNN = resnet18(cnn_layers=CNN_LAYERS, in_lead=1).to(DEVICE)
DROPOUT = 0.01  # dropout probability
model = TransformerModel(CLASS, EMB_SIZE, nHEADS, D_HID, nLAYERS, modelCNN, DROPOUT).to(DEVICE)

model = nn.DataParallel(model)
