from FCNN import Net
from utils import *


class DeviceManager:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"device available : {self.device}")

    def __call__(self):
        return self.device


class NetManager:

    def __init__(self, model_weights_path):
        self.device = DeviceManager()
        self.net = Net()
        self.net.eval()
        self.net.to(self.device())
        self.model_weights_path = model_weights_path

    def load_weights(self):
        flag = self.net.load_state_dict(torch.load(self.model_weights_path, map_location=torch.device(self.device())))
        print(flag)

    def create_checkpoint(self):
        torch.save(self.net.state_dict(), self.model_weights_path)

    def preprocess_(self, input_frame, background_image, resize_to=224, normalize=False):

        # resizing
        input_frame = cv2.resize(input_frame, (resize_to, resize_to))
        background_image = cv2.resize(background_image, (resize_to, resize_to))

        # normalise
        if normalize:
            input_frame /= 255.0
            background_image /= 255.0

        return input_frame, background_image

    def preprocess(self, input_frame, background_image):

        assert input_frame.shape == background_image.shape, "shape mismatch"

        # change to channel first
        input_frame, background_image = channel_first(input_frame), channel_first(background_image)

        # convert to tensor
        input_frame = torch.tensor(input_frame)
        background_image = torch.tensor(background_image)

        # concat bg and input
        hybrid = torch.cat([background_image, input_frame], 0)

        assert hybrid.shape[0] == 6, "not a 6 channel hybrid"

        # make batch
        hybrid = hybrid.unsqueeze(dim=0)
        return hybrid

    def postprocess(self, output):

        if output.ndim == 3:
            output = output.squeeze(dim=0)

        assert output.ndim == 2, "2 dimension mask is expectected from neural network. Function written for single " \
                                 "batch only. "
        output = output.cpu().numpy()

        # median filter
        # output = (cv2.medianBlur(np.uint8(output * 255), 9) / 255.0)

        # threshold
        print(f"outputs range=>>> : {np.min(output), np.max(output)}")
        output[output >= 0.8] = 1
        output[output < 0.8] = 0

        return np.uint8(output)

    @torch.no_grad()
    def __call__(self, input_frame, background_image):
        """ forwards the frame and background through net """
        assert input_frame.ndim == 3 and background_image.ndim == 3, "colored 3 channel input and background required."
        inputs = self.preprocess(input_frame, background_image)
        # inputs = inputs/255.0
        inputs.to(self.device())
        output = self.net(inputs)
        output = self.postprocess(output)
        print(f'''
                hybrid input shape : {inputs.shape}
                output shape : {output.shape}
                
                inputs_unique_values : {torch.unique(inputs)}
                output_unique_values : {np.unique(output, return_counts=True)}
                
                inputs range : {torch.min(inputs), torch.max(inputs)}
                outputs range : {np.min(output), np.max(output)}
                
               ''')

        return output

    def __str__(self):
        return str(self.net)


def main(camera_on=False):
    weights_path = "model_weights/weights-net-7.pt"
    net = NetManager(weights_path)
    net.load_weights()

    background_image = "sample_inputs/2020-12-20-001927.jpg"
    background_image = np.float32(read_image(background_image))

    if camera_on:
        camera = cv2.VideoCapture(0)
        while True:
            cv2.imshow("My background", np.uint8(background_image))
            ret, frame = camera.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mask = net(np.float32(frame), background_image)

            print(f"the output mask unique values : {np.unique(mask, return_counts=True)}")
            cv2.imshow("My cam video", frame)
            cv2.imshow("My mask", mask * 255)

            # Close and break the loop after pressing "x" key
            if cv2.waitKey(1) & 0XFF == ord('x'):
                break

            background_image *= 255.0
            # input("press enter...")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    weights_path = "model_weights/weights-net-7.pt"
    # sample_bg_image = "sample_inputs/back_1.jpg"
    # sample_input_image = "sample_inputs/in_1.jpg"

    sample_bg_image = "sample_inputs/back2.jpg"
    sample_input_image = "sample_inputs/in2.jpg"

    # inputs = {sample_bg_image: sample_input_image}

    inputs = {'bg_office.jpg': ["input_office.jpg"], 'bg_highway.jpg': ['input_highway.jpg'],
              "bg_sofa.jpg": ['input_sofa1.jpg', 'input_sofa2.jpg']}

    net = NetManager(weights_path)
    net.load_weights()
    print(net)

    for k, v in inputs.items():
        # test input
        sample_input_image = os.path.join("sample_inputs", v[0])
        sample_bg_image = os.path.join("sample_inputs", k)

        if not (os.path.exists(sample_input_image) or os.path.exists(sample_bg_image)):
            continue

        # read image
        input_image = read_image(sample_input_image)
        bg_image = read_image(sample_bg_image)

        # preprocess them
        input_image, bg_image = net.preprocess_(input_image, bg_image, resize_to=224, normalize=True)

        fig, axes = plt.subplots(1, 3)
        axes[0].imshow(input_image);
        axes[0].set_title("in")
        axes[1].imshow(bg_image);
        axes[1].set_title("bg")

        # forward
        predicted_mask = net(input_image, bg_image)
        # predicted_mask = predicted_mask

        axes[2].imshow(predicted_mask)
        axes[2].set_title("mask")

        if not os.path.exists("./outputs"):
            os.mkdir("./outputs")
        plt.savefig(os.path.join("./outputs/out_") + k)

        plt.show()

    # main(camera_on=True)
