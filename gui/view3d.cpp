/****************************************************************************

 Copyright (C) 2002-2008 Gilles Debunne. All rights reserved.

 This file is part of the QGLViewer library version 2.3.6.

 http://www.libqglviewer.com - contact@libqglviewer.com

 This file may be used under the terms of the GNU General Public License 
 versions 2.0 or 3.0 as published by the Free Software Foundation and
 appearing in the LICENSE file included in the packaging of this file.
 In addition, as a special exception, Gilles Debunne gives you certain 
 additional rights, described in the file GPL_EXCEPTION in this package.

 libQGLViewer uses dual licensing. Commercial/proprietary software must
 purchase a libQGLViewer Commercial License.

 This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
 WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

*****************************************************************************/

#include "view3d.h"

#define GL_GLEXT_PROTOTYPES
#include <GL/glext.h>
extern "C" {
	extern void APIENTRY glPointParameterfv (GLenum, const GLfloat *);
}

using namespace std;

void View3D::setColor(float h, float a)
{
	h = 0.8f*(0.8f - h);
	int i;
	float aa, bb, cc, f, r, g, b;

	h *= 6.0;
	i = floorf(h);
	f = h - i;
	bb = 1 - f;
	cc = f;
	switch (i) {
		case 0: r = 1.f; g = cc;  b = aa; break;
		case 1: r = bb;  g = 1.f; b = aa; break;
		case 2: r = aa;  g = 1.f; b = cc; break;
		case 3: r = aa;  g = bb;  b = 1.f;break;
		case 4: r = cc;  g = aa;  b = 1.f;break;
		case 5: r = 1.f; g = aa;  b = bb; break;
    }
    glColor4f(0.25f*r, 0.25f*g, 0.25f*b, 0.05f*a);
}


void View3D::draw()
{
	/*	const float nbSteps = 200.0;

	glBegin(GL_QUAD_STRIP);
	for (int i=0; i<nbSteps; ++i)
	{
		const float ratio = i/nbSteps;
		const float angle = 21.0*ratio;
		const float c = cos(angle);
		const float s = sin(angle);
		const float r1 = 1.0 - 0.8f*ratio;
		const float r2 = 0.8f - 0.8f*ratio;
		const float alt = ratio - 0.5f;
		const float nor = 0.5f;
		const float up = sqrt(1.0-nor*nor);
		glColor3f(1.0-ratio, 0.2f , ratio);
		glNormal3f(nor*c, up, nor*s);
		glVertex3f(r1*c, alt, r1*s);
		glVertex3f(r2*c, alt+0.05f, r2*s);
	}
	glEnd();*/
	
	float h = source.height, w = source.width, l = source.size();
	
	glPushMatrix();
	glRotatef(90.f, 0.f, 0.f, -1.f);
	glTranslatef(-0.5, -0.5, -0.5);
	glScalef(1.f/h, 1.f/w, 0.25f/l);
	for (int y = 0; y < source.height; ++y) {
		glPushMatrix();
	  	for (int x = 0; x < source.width; ++x) {
			glPushMatrix();
	  		const multi_img::Pixel& p = source(y, x);
		  	for (int d = p.size()-1; d >= 0; --d) {
		  		if (p[d] > 8.f) {
					setColor((float)d/l, p[d]/255.0f);
					glCallList(dlCube);
				}
		  		glTranslatef(0, 0, 1);
			}
			glPopMatrix();
	  		glTranslatef(0, 1, 0);
	  	}
		glPopMatrix();
  		glTranslatef(1, 0, 0);
  	}
	glPopMatrix();
}

void View3D::init()
{
//	glEnable(GL_CULL_FACE);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 

/*	glEnable(GL_POINT_SMOOTH);
	glPointSize(5.f);
	GLfloat att[] = { 1., 1., 2. };
	glPointParameterfv(GL_POINT_DISTANCE_ATTENUATION, att);*/

	// build cube
	dlCube = glGenLists(1);
	glNewList(dlCube, GL_COMPILE);
/*	glBegin(GL_POINTS);
		glVertex3f(0.0f, 0.0f, 0.0f);*/
	glBegin(GL_QUADS);
		glVertex3f( 0.5f, 0.5f, 0.5f);			// Top Right Of The Quad (Front)
		glVertex3f(-0.5f, 0.5f, 0.5f);			// Top Left Of The Quad (Front)
		glVertex3f(-0.5f,-0.5f, 0.5f);			// Bottom Left Of The Quad (Front)
		glVertex3f( 0.5f,-0.5f, 0.5f);			// Bottom Right Of The Quad (Front)
/*		glVertex3f( 0.5f, 0.5f,-0.5f);			// Top Right Of The Quad (Top)
		glVertex3f(-0.5f, 0.5f,-0.5f);			// Top Left Of The Quad (Top)
		glVertex3f(-0.5f, 0.5f, 0.5f);			// Bottom Left Of The Quad (Top)
		glVertex3f( 0.5f, 0.5f, 0.5f);			// Bottom Right Of The Quad (Top)
		glVertex3f( 0.5f,-0.5f,-0.5f);			// Bottom Left Of The Quad (Back)
		glVertex3f(-0.5f,-0.5f,-0.5f);			// Bottom Right Of The Quad (Back)
		glVertex3f(-0.5f, 0.5f,-0.5f);			// Top Right Of The Quad (Back)
		glVertex3f( 0.5f, 0.5f,-0.5f);			// Top Left Of The Quad (Back)
		glVertex3f(-0.5f, 0.5f, 0.5f);			// Top Right Of The Quad (Left)
		glVertex3f(-0.5f, 0.5f,-0.5f);			// Top Left Of The Quad (Left)
		glVertex3f(-0.5f,-0.5f,-0.5f);			// Bottom Left Of The Quad (Left)
		glVertex3f(-0.5f,-0.5f, 0.5f);			// Bottom Right Of The Quad (Left)
		glVertex3f( 0.5f, 0.5f,-0.5f);			// Top Right Of The Quad (Right)
		glVertex3f( 0.5f, 0.5f, 0.5f);			// Top Left Of The Quad (Right)
		glVertex3f( 0.5f,-0.5f, 0.5f);			// Bottom Left Of The Quad (Right)
		glVertex3f( 0.5f,-0.5f,-0.5f);			// Bottom Right Of The Quad (Right)*/
	glEnd();
	glEndList();

	source.rebuildPixels();

	// Restore previous viewer state.
	restoreStateFromFile();
}

QString View3D::helpString() const
{
  QString text("<h2>3D Multispectral Image Viewer</h2>");
	text += "Use the mouse to move the camera around the object. "
	"You can respectively revolve around, zoom and translate with the three mouse buttons. "
	"Left and middle buttons pressed together rotate around the camera view direction axis<br><br>"
	"Pressing <b>Alt</b> and one of the function keys (<b>F1</b>..<b>F12</b>) defines a camera keyFrame. "
	"Simply press the function key again to restore it. Several keyFrames define a "
	"camera path. Paths are saved when you quit the application and restored at next start.<br><br>"
	"Press <b>F</b> to display the frame rate, <b>A</b> for the world axis, "
	"<b>Alt+Return</b> for full screen mode and <b>Control+S</b> to save a snapshot. "
	"See the <b>Keyboard</b> tab in this window for a complete shortcut list.<br><br>"
	"Double clicks automates single click actions: A left button double click aligns the closer axis with the camera (if close enough). "
	"A middle button double click fits the zoom of the camera and the right button re-centers the scene.<br><br>"
	"A left button double click while holding right button pressed defines the camera <i>Revolve Around Point</i>. "
	"See the <b>Mouse</b> tab and the documentation web pages for details.<br><br>"
	"Press <b>Escape</b> to exit the viewer.";
  return text;
}
