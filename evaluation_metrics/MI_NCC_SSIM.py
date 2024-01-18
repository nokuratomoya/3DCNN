import calcMI_main
import NCC_main
import SSIM


def main():
    calcMI_main.main()
    print("MI finished")
    NCC_main.main()
    print("NCC finished")
    SSIM.main()
    print("SSIM finished")


if __name__ == "__main__":
    main()
