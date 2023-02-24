mkdir kodak_vis_png;
cd kodak_vis_png;

mkdir kodak_vis_png_22;
mkdir kodak_vis_png_27;
mkdir kodak_vis_png_32;
mkdir kodak_vis_png_37;


cd ..;

ffmpeg -i kodak_vis_qp22.bin ./kodak_vis_png/kodak_vis_png_22/%d.png;
ffmpeg -i kodak_vis_qp27.bin ./kodak_vis_png/kodak_vis_png_27/%d.png;
ffmpeg -i kodak_vis_qp32.bin ./kodak_vis_png/kodak_vis_png_32/%d.png;
ffmpeg -i kodak_vis_qp37.bin ./kodak_vis_png/kodak_vis_png_37/%d.png;


