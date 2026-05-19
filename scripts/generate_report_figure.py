from utils.visualize import generate_skip_connection_figure, generate_skip_connection_figure_3d


def main():
    save_path_2d = generate_skip_connection_figure("outputs/figures")
    save_path_3d = generate_skip_connection_figure_3d("outputs/figures")
    print(f"saved 2d figure to: {save_path_2d}")
    print(f"saved 3d figure to: {save_path_3d}")


if __name__ == "__main__":
    main()
