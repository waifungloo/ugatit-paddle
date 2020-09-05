import os

def make_a_txt(filename, txtpath):
    filePath = 'D:/Desktop/paddle/Cycle_GAN_paddle/data/selfie2anime'
    with open(txtpath, 'w') as f:
        for file in os.listdir(os.path.join(filePath,filename)):
            file_path = filename + '/' + file
            f.write(file_path)
            f.write('\n')


if __name__ == '__main__':
    all_txt = ["trainA", "trainB", "testA", "testB"]
    for t in all_txt:
        make_a_txt(t, os.path.join("D:/Desktop/paddle/Cycle_GAN_paddle/data/selfie2anime", (t + ".txt")))