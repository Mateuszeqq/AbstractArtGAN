import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2


latent_dim = 100


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.n = 512
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_dim, out_channels=self.n, kernel_size=(4, 4), bias=False),
            nn.BatchNorm2d(self.n),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=self.n, out_channels=self.n // 2, kernel_size=(4, 4), stride =(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(self.n // 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=self.n // 2, out_channels=self.n // 4, kernel_size=(4, 4), stride =(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(self.n // 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=self.n // 4, out_channels=self.n // 8, kernel_size=(4, 4), stride =(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(self.n // 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=self.n // 8, out_channels=3, kernel_size=(4, 4), stride =(2, 2), padding=(1, 1), bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


@st.cache
def load_generator():
    generator = Generator(latent_dim=latent_dim)
    generator.load_state_dict(torch.load('generator.pt'))
    return generator


def generate_abstract_art(generator):
    means = torch.Tensor((0.5, 0.5, 0.5))
    stds = torch.Tensor((0.5, 0.5, 0.5))
    unnormalize = transforms.Normalize((-1*means / stds).tolist(), (1.0 / stds).tolist())
    noise = torch.randn(1, latent_dim, 1, 1)
    return unnormalize(generator(noise))


def main():
    st.set_page_config(layout="centered", page_icon="🎨", page_title="Abstract Art Generator")
    st.title("🎨 Abstract Art Generator")

    generator = load_generator()

    left, right = st.columns(2)

    right.write("Generated art:")

    left.write("Fill in the data:")
    form = left.form("template_form")
    file_name = form.text_input("File name")

    submit = form.form_submit_button("Generate Abstract Art!")

    if submit:
        generated_img = generate_abstract_art(generator).detach().numpy()
        generated_img = np.moveaxis(generated_img[0], 0, -1)

        right.image(generated_img, clamp=True, channels='RGB')
        generated_img = generated_img[:, :, ::-1]

        right.success("🎉 Your art was generated!")
        if file_name != '':
            generated_img = generated_img * 255
            _, JPEG = cv2.imencode('.jpeg', generated_img)
            right.download_button(
                "⬇️ Download Art",
                data=JPEG.tobytes(),
                file_name=f"{file_name}.jpeg",
                mime="application/octet-stream",
            )


if __name__ == '__main__':
    main()
