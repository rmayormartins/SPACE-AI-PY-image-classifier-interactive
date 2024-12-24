import os
import shutil
import joblib  
import gradio as gr
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
from sklearn.metrics import classification_report, confusion_matrix


os.system('apt-get update')
os.system('apt-get install -y protobuf-compiler')


model_dict = {
    'AlexNet': models.alexnet,
    'ResNet18': models.resnet18,
    'ResNet34': models.resnet34,
    'ResNet50': models.resnet50,
    'MobileNetV2': models.mobilenet_v2
}

#
model = None
train_loader = None
val_loader = None
test_loader = None
dataset_path = 'dataset'
class_dirs = []
test_dataset_path = 'test_dataset'
test_class_dirs = []
num_classes = 2  # 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 
def setup_classes(num_classes_value):
    global class_dirs, dataset_path, num_classes

    num_classes = int(num_classes_value)  # 
    # 
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    os.makedirs(dataset_path)

    # 
    class_dirs = [os.path.join(dataset_path, f'class_{i}') for i in range(num_classes)]
    for class_dir in class_dirs:
        os.makedirs(class_dir)

    return f"Criados {num_classes} diretórios para classes."

# 
def upload_images(class_id, images):
    class_dir = class_dirs[int(class_id)]
    for image in images:
        shutil.copy(image, class_dir)
    return f"Imagens salvas na classe {class_id}."

# 
def prepare_data(batch_size=32, resize=(224, 224)):
    global train_loader, val_loader, test_loader, num_classes

    # 
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(dataset_path, transform=transform)

    if len(dataset.classes) != num_classes:
        return f"Erro: Número de classes detectadas ({len(dataset.classes)}) não corresponde ao número esperado ({num_classes}). Verifique suas imagens."

    # 
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return "Preparação dos dados concluída com sucesso."

# 
def start_training(model_name, epochs, lr):
    global model, train_loader, val_loader, device

    if train_loader is None or val_loader is None:
        return "Erro: Dados não preparados."

    model = model_dict[model_name](pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(lr))

    for epoch in range(int(epochs)):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

    torch.save(model.state_dict(), 'modelo.pth')
    return f"Treinamento concluído com sucesso. Modelo salvo."

# 
def evaluate_model(loader):
    global model, device, num_classes

    if model is None:
        return "Erro: Modelo não treinado."

    if loader is None:
        return "Erro: Conjunto de dados de teste não está preparado."

    model.eval()
    all_preds = []
    all_labels = []
    try:
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        report = classification_report(all_labels, all_preds, labels=list(range(num_classes)), target_names=[f"class_{i}" for i in range(num_classes)], zero_division=0)
        return report
    except Exception as e:
        return f"Erro durante a avaliação: {str(e)}"

# 
def show_confusion_matrix(loader):
    global model, device, num_classes

    if model is None:
        return "Erro: Modelo não treinado."

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    plt.figure(figsize=(6, 4.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[f"class_{i}" for i in range(num_classes)], yticklabels=[f"class_{i}" for i in range(num_classes)])
    plt.xlabel('Predictions')
    plt.ylabel('Actuals')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return Image.open(buf)

# 
def predict_images(images):
    global model, device, num_classes

    if model is None:
        return "Erro: Modelo não treinado."

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    model.eval()
    results = []

    for image in images:
        try:
            img = transform(Image.open(image)).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(img)
                _, preds = torch.max(outputs, 1)
                predicted_class = preds.item()
                results.append(f"Imagem {os.path.basename(image)} - Classe prevista: class_{predicted_class}")
        except Exception as e:
            results.append(f"Erro ao processar a imagem {image}: {str(e)}")

    return results

# 
def export_model(format):
    global model

    if model is None:
        return "Erro: Modelo não treinado."

    file_path = f"modelo_exportado.{format}"
    if format == "pth":
        torch.save(model.state_dict(), file_path)
    elif format == "onnx":
        try:
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            torch.onnx.export(model, dummy_input, file_path, export_params=True, opset_version=10, input_names=['input'], output_names=['output'])
        except Exception as e:
            return f"Erro ao exportar para ONNX: {str(e)}"
    elif format == "pkl":
        joblib.dump(model, file_path)
    else:
        return f"Formato {format} não suportado."

    return f"Modelo exportado com sucesso para {file_path}"

# 
def setup_test_classes():
    global test_class_dirs, test_dataset_path

    if os.path.exists(test_dataset_path):
        shutil.rmtree(test_dataset_path)
    os.makedirs(test_dataset_path)

    # 
    test_class_dirs = [os.path.join(test_dataset_path, f'class_{i}') for i in range(num_classes)]
    for class_dir in test_class_dirs:
        os.makedirs(class_dir)

    return f"Criados {num_classes} diretórios para classes de teste."

# 
def upload_test_images(class_id, images):
    class_dir = test_class_dirs[int(class_id)]
    for image in images:
        shutil.copy(image, class_dir)
    return f"Imagens de teste salvas na classe {class_id}."

# 
def prepare_test_data(batch_size=32, resize=(224, 224)):
    global test_loader, num_classes

    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
    ])

    test_dataset = datasets.ImageFolder(test_dataset_path, transform=transform)

    if len(test_dataset.classes) != num_classes:
        return f"Erro: Número de classes detectadas ({len(test_dataset.classes)}) não corresponde ao número esperado ({num_classes}). Verifique suas imagens."

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return "Preparação dos dados de teste concluída com sucesso."

# 
def main():
    with gr.Blocks() as demo:
        gr.Markdown("# Image Classification Training")

        with gr.Tab("Configurar Classes"):
            num_classes_input = gr.Number(label="Número de Classes", value=2, precision=0)
            setup_button = gr.Button("Configurar Classes")
            setup_output = gr.Textbox()
            setup_button.click(setup_classes, inputs=num_classes_input, outputs=setup_output)

        with gr.Tab("Upload de Imagens"):
            upload_inputs = []
            for i in range(num_classes):
                with gr.Column():
                    gr.Markdown(f"### Classe {i}")
                    class_id = gr.Number(label=f"ID da Classe {i}", value=i, precision=0)
                    images = gr.File(label="Upload de Imagens", file_count="multiple", type="filepath")
                    upload_button = gr.Button("Upload")
                    upload_output = gr.Textbox()

                    upload_inputs.append((class_id, images, upload_button, upload_output))
                    upload_button.click(upload_images, inputs=[class_id, images], outputs=upload_output)

        with gr.Tab("Preparação de Dados"):
            batch_size = gr.Number(label="Tamanho do Batch", value=32)
            resize = gr.Textbox(label="Resize (Ex: 224,224)", value="224,224")
            prepare_button = gr.Button("Preparar Dados")
            prepare_output = gr.Textbox()
            prepare_button.click(lambda batch_size, resize: prepare_data(batch_size=batch_size, resize=tuple(map(int, resize.split(',')))), inputs=[batch_size, resize], outputs=prepare_output)

        with gr.Tab("Treinamento"):
            model_name = gr.Dropdown(label="Modelo", choices=list(model_dict.keys()))
            epochs = gr.Number(label="Épocas", value=30)
            lr = gr.Number(label="Taxa de Aprendizado", value=0.001)
            train_button = gr.Button("Iniciar Treinamento")
            train_output = gr.Textbox()
            train_button.click(start_training, inputs=[model_name, epochs, lr], outputs=train_output)

        with gr.Tab("Avaliação do Modelo"):
            eval_button = gr.Button("Avaliar Modelo")
            eval_output = gr.Textbox()
            eval_button.click(lambda: evaluate_model(test_loader), outputs=eval_output)

            cm_button = gr.Button("Mostrar Matriz de Confusão")
            cm_output = gr.Image()
            cm_button.click(lambda: show_confusion_matrix(test_loader), outputs=cm_output)

        with gr.Tab("Predição e Avaliação"):
            predict_images_input = gr.File(label="Upload de Imagens para Predição", file_count="multiple", type="filepath")
            predict_button = gr.Button("Predizer")
            predict_output = gr.Textbox()
            predict_button.click(predict_images, inputs=predict_images_input, outputs=predict_output)

            gr.Markdown("### Upload de Imagens de Teste")
            setup_test_button = gr.Button("Configurar Diretórios de Teste")
            setup_test_output = gr.Textbox()
            setup_test_button.click(setup_test_classes, outputs=setup_test_output)

            upload_test_inputs = []
            for i in range(num_classes):
                with gr.Column():
                    gr.Markdown(f"### Classe de Teste {i}")
                    test_class_id = gr.Number(label=f"ID da Classe {i}", value=i, precision=0)
                    test_images = gr.File(label="Upload de Imagens de Teste", file_count="multiple", type="filepath")
                    upload_test_button = gr.Button("Upload Imagens de Teste")
                    upload_test_output = gr.Textbox()

                    upload_test_inputs.append((test_class_id, test_images, upload_test_button, upload_test_output))
                    upload_test_button.click(upload_test_images, inputs=[test_class_id, test_images], outputs=upload_test_output)

            prepare_test_button = gr.Button("Preparar Dados de Teste")
            prepare_test_output = gr.Textbox()
            prepare_test_button.click(lambda batch_size, resize: prepare_test_data(batch_size=batch_size, resize=tuple(map(int, resize.split(',')))), inputs=[batch_size, resize], outputs=prepare_test_output)

            eval_test_button = gr.Button("Avaliar Conjunto de Teste")
            eval_test_output = gr.Textbox()
            eval_test_button.click(lambda: evaluate_model(test_loader), outputs=eval_test_output)

            cm_test_button = gr.Button("Mostrar Matriz de Confusão do Conjunto de Teste")
            cm_test_output = gr.Image()
            cm_test_button.click(lambda: show_confusion_matrix(test_loader), outputs=cm_test_output)

        with gr.Tab("Exportação"):
            export_format = gr.Radio(label="Formato", choices=["pth", "onnx", "pkl"])
            export_button = gr.Button("Exportar Modelo")
            export_output = gr.Textbox()
            export_button.click(export_model, inputs=export_format, outputs=export_output)

    demo.launch()

if __name__ == "__main__":
    main()
