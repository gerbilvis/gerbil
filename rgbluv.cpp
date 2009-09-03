#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void rgb2hsv(float r, float g, float b, float *H, float *S, float *V) {
	float maxv = r > g ? r : g;
	if (b > maxv)
		maxv = b;
	*V = maxv;
	if (maxv > 0) {
		float minv = r < g ? r : g;
		if (b < minv)
			minv = b;
		*S = 1.0 - minv / maxv;
		if (maxv > minv) {
			if (maxv == r) {
				*H = (g - b) / (maxv - minv);
				if (*H < 0)
					*H += 6.0;
			} else if (maxv == g)
				*H = 2.0 + (b - r) / (maxv - minv);
			else
				*H = 4.0 + (r - g) / (maxv - minv);
		}
	}
}

void rgb2luv(float *RGB, float *LUV, int size) {
	int    i;
	double x, y, X, Y, Z, den, u2, v2, X0, Z0, Y0, u20, v20, r, g, b;

	X0 = (0.607 + 0.174 + 0.201);
	Y0 = (0.299 + 0.587 + 0.114);
	Z0 = (0.066 + 1.117);

/* Y0 = 1.0 */
	u20 = 4 * X0 / (X0 + 15 * Y0 + 3 * Z0);
	v20 = 9 * Y0 / (X0 + 15 * Y0 + 3 * Z0);

	for (i = 0; i < size; i += 3) {
		if (RGB[i] <= 20.)
			r = (double)(8.715e-4 * RGB[i]);
		else
			r = (double)pow((RGB[i] + 25.245) / 280.245, 2.22);

		if (RGB[i + 1] <= 20.)
			g = (double)(8.715e-4 * RGB[i + 1]);
		else
			g = (double)pow((RGB[i + 1] + 25.245) / 280.245, 2.22);

		if (RGB[i + 2] <= 20.)
			b = (double)(8.715e-4 * RGB[i + 2]);
		else
			b = (double)pow((RGB[i + 2] + 25.245) / 280.245, 2.22);

		X = 0.412453 * r + 0.357580 * g + 0.180423 * b;
		Y = 0.212671 * r + 0.715160 * g + 0.072169 * b;
		Z = 0.019334 * r + 0.119193 * g + 0.950227 * b;

		if (X == 0.0 && Y == 0.0 && Z == 0.0) {
			x = 1.0 / 3.0; y = 1.0 / 3.0;
		} else {
			den = X + Y + Z;
			x   = X / den; y = Y / den;
		}

		den = -2 * x + 12 * y + 3;
		u2  = 4 * x / den;
		v2  = 9 * y / den;

		if (Y > 0.008856)
			LUV[i] = (float)(116 * pow(Y, 1.0 / 3.0) - 16);
		else
			LUV[i] = (float)(903.3 * Y);
		LUV[i + 1] = (float)(13 * LUV[i] * (u2 - u20));
		LUV[i + 2] = (float)(13 * LUV[i] * (v2 - v20));
	}
}

void luv2rgb(float *RGB, float *LUV, int size) {
	int    i, k;
	double x, y, X, Y, Z, den, u2, v2, X0, Z0, Y0, u20, v20, vec[3];

	X0 = (0.607 + 0.174 + 0.201);
	Y0 = (0.299 + 0.587 + 0.114);
	Z0 = (0.066 + 1.117);

/* Y0 = 1.0 */
	u20 = 4 * X0 / (X0 + 15 * Y0 + 3 * Z0);
	v20 = 9 * Y0 / (X0 + 15 * Y0 + 3 * Z0);

	for (i = 0; i < size; i += 3) {
		if (LUV[i] > 0) {
			if (LUV[i] < 8.0)
				Y = ((double)LUV[i]) / 903.3;
			else
				Y = pow((((double)LUV[i]) + 16) / 116.0, 3.0);
			u2 =
				((double)LUV[i +
							 1]) / 13.0 / ((double)LUV[i]) +
				u20;
			v2 =
				((double)LUV[i +
							 2]) / 13.0 / ((double)LUV[i]) +
				v20;

			den = 6 + 3 * u2 - 8 * v2;
			if (den < 0)
				printf("den<0\n");
			if (den == 0)
				printf("den==0\n");
			x = 4.5 * u2 / den;
			y = 2.0 * v2 / den;

			X = x / y * Y;
			Z = (1 - x - y) / y * Y;
		} else { X = 0.0; Y = 0.0; Z = 0.0; }

		vec[0] = (3.240479 * X - 1.537150 * Y - 0.498536 * Z);
		vec[1] = (-0.969256 * X + 1.875992 * Y + 0.041556 * Z);
		vec[2] = (0.055648 * X - 0.204043 * Y + 1.057311 * Z);
		for (k = 0; k < 3; k++) {
			if (vec[k] <= 0.018)
				vec[k] = 255 * 4.5 * vec[k];
			else
				vec[k] = 255 *
						 (1.099 * pow(vec[k], 0.45) - 0.099);
			if (vec[k] > 255)
				vec[k] = 255;
			else if (vec[k] < 0)
				vec[k] = 0;
			RGB[i + k] = roundf((float)vec[k]);
		}
	}
}
