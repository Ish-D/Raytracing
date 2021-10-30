#include <iostream>
using namespace std;

int main()
{
    // Canvas

    const int image_width = 256;
    const int image_height = 256;

    // Render 
    cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    // Pixels written out from left to right (i = 0)
    // Rows top to bottom (j = image_height-1)
    for (int j = image_height-1; j>= 0; --j)
    {
        // Progress Bar
        cerr << "\rScanlines remaining: " << j << ' ' << flush;
        for (int i = 0; i<image_width; ++i)
        {
            auto r = double(i)/(image_width-1);
            auto g = double(j)/(image_height-1);
            auto b = 0.25;

            int ir = static_cast<int>(255.999*r);
            int ig = static_cast<int>(255.999*g);
            int ib = static_cast<int>(255.999*b);

            cout << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }
    cerr << "\nDone.";
}