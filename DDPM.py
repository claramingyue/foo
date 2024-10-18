import os
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from pathlib import Path
import torch
import matplotlib.pyplot as plt

# Configuración del modelo U-Net
def crear_modelo_unet():
    return Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=True
    )

# Configuración de la difusión
def crear_diffusion(model):
    return GaussianDiffusion(
        model,
        image_size=256,
        timesteps=1000,           # número de pasos de entrenamiento
        sampling_timesteps=250    # pasos de muestreo (usando DDIM para inferencia rápida)
    )

# Guardar el modelo una vez entrenado
def guardar_modelo(diffusion, model_path):
    torch.save(diffusion.state_dict(), model_path)
    print(f"Modelo guardado en {model_path}")

# # Cargar el modelo guardado
# def cargar_modelo(model_path, model):
#     loaded_diffusion = GaussianDiffusion(
#         model,
#         image_size=256,
#         timesteps=1000,
#         sampling_timesteps=250
#     )
#     loaded_diffusion.load_state_dict(torch.load(model_path))
#     print(f"Modelo cargado desde {model_path}")
#     return loaded_diffusion

# # Generar imágenes
# def generar_imagenes(diffusion, output_dir, num_images_to_generate=100):
#     os.makedirs(output_dir, exist_ok=True)
#     images_generated = 0

#     # Mover el modelo a la GPU si está disponible
#     if torch.cuda.is_available():
#         diffusion.to('cuda')

#     while images_generated < num_images_to_generate:
#         batch_images = diffusion.sample(batch_size=10)  # Cambia el tamaño del lote si tu GPU lo permite
#         for i in range(batch_images.shape[0]):
#             img = batch_images[i].permute(1, 2, 0).cpu().numpy()  # Cambiar formato de tensor a imagen
#             fpath = f'AAimage_{images_generated}-{images_generated + i}.png'
#             img_path = os.path.join(output_dir, fpath)
#             plt.imsave(img_path, img)
#             print(f"Generated: {img_path}")
#         images_generated += batch_images.shape[0]

#     print(f"Imágenes generadas y guardadas en {output_dir}")

def main():
    # Definir rutas
    # dataset_path = Path('/content/drive/MyDrive/dataset_preprocesado_png')
    # model_path = '/content/drive/MyDrive/trained_diffusion_model_png.pth'
    # output_dir = '/content/drive/MyDrive/Colab_Notebooks/SynteticImageAnalysis/DDPM_results_png'
    dataset_path = Path('/Neuronal/bedroom_dataset_preprocesado256')
    model_path = '/home/claramingyue/DDPM/modelo_jpg'
    #output_dir = '/home/claramingyue/DDPM/SynteticImageAnalysis/DDPM_results_png'

    # Verificar si la carpeta del dataset existe
    if not dataset_path.exists():
        raise FileNotFoundError(f"La carpeta del dataset no se encuentra: {dataset_path}")

    # Crear modelo U-Net y difusión
    model = crear_modelo_unet()
    diffusion = crear_diffusion(model)

    # Configurar y entrenar el modelo
    trainer = Trainer(
        diffusion,
        str(dataset_path),
        train_batch_size=16,
        train_lr=8e-5,
        train_num_steps=10000,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        amp=True,
        calculate_fid=False
    )
    
    # Iniciar el entrenamiento
    trainer.train()

    # Guardar el modelo entrenado
    guardar_modelo(diffusion, model_path)


if __name__ == "__main__":
    main()

