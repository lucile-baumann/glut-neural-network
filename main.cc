// Fichier principal de l'utilisation du réseau de neurones

#include <algorithm>
#include <time.h>
#include <sys/types.h>
#include <GL/freeglut.h>
#include <string>

#include "reseau.h"

static int iteration;
static Reseau Network;
static Type currentType;
static int GlobalSuccess;
static int Success3[100];

static void Line(int x1, int y1, int x2, int y2)
{
    glBegin(GL_LINES);
    glVertex2f(x1, y1);
    glVertex2f(x2, y2);
    glEnd();
}

static void Rectangle(int x1, int y1, int x2, int y2, bool rempli)
{
    if (rempli)
        glBegin(GL_POLYGON);
    else
        glBegin(GL_LINE_LOOP);
    glVertex2f(x1, y1);
    glVertex2f(x2, y1);
    glVertex2f(x2, y2);
    glVertex2f(x1, y2);
    glEnd();
}

static void displayText(std::string text)
{
    glutBitmapString(GLUT_BITMAP_8_BY_13, (const unsigned char*)text.c_str());
}

static void display_input()
{
    glRasterPos2f(80, 70);
    displayText("Input Layer");

    for (unsigned u = 0; u < Network.getLayerSize(Reseau::Layer::Input); ++u)
    {
        auto state = Network.getLayer(Reseau::Layer::Input).at(u).etat;
        auto gray_level = (state + Network.getNoise() / 2.0) / (1 + Network.getNoise());
        glColor3f(gray_level, gray_level, gray_level);

        auto x = (u % 16) * 5 + 80;
        auto y = (u / 16) * 5 + 80;
        Rectangle(x, y, x + 5, y + 5, true);
    }
}

std::string TypeToString(const Type& type)
{
    switch (type)
    {
    case Type::CheckedPattern: return "Checked Pattern";
    case Type::Cross: return "Cross";
    case Type::HorizontalLine: return "Horizontal Line";
    case Type::VerticalLine: return "Vertical Line";
    case Type::Square: return "Square";
    case Type::Triangle: return "Triangle";
    }
}
static void display()
{
    // White background
    glClearColor(1,1,1, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    // Black pencil
    glColor3f(0, 0, 0);

    Line(20, 2, 980, 2);
    Line(20, 45, 980, 45);

    Line(220, 70, 220, 480);
    Line(680, 70, 680, 480);

    const float line1 = 20.f;
    const float line2 = 40.f;
    float pos = 40.f;

    glRasterPos2f(pos, line1);
    displayText("Iteration");
    glRasterPos2f(pos, line2);
    displayText(std::to_string(iteration));

    pos += 150.f;
    glRasterPos2f(pos, line1);
    displayText("Epsilon");
    glRasterPos2f(pos, line2);
    displayText(std::to_string(EPSILON));

    pos += 150.f;
    glRasterPos2f(pos, line1);
    displayText("Bruit");
    glRasterPos2f(pos, line2);
    displayText(std::to_string(Network.getNoise()));

    pos += 150.f;
    glRasterPos2f(pos, line1);
    displayText("Input");
    glRasterPos2f(pos, line2);
    displayText(TypeToString(currentType));

    pos += 150.f;
    glRasterPos2f(pos, line1);
    displayText("Global success");
    glRasterPos2f(pos, line2);
    displayText(std::to_string(GlobalSuccess * 100.0 / iteration) + "%");

    pos += 150.f;
    glRasterPos2f(pos, line1);
    displayText("Instant success");
    glRasterPos2f(pos, line2);
    displayText(std::to_string(Success3[iteration % 100]) + "%");

    display_input();

    glRasterPos2f(420, 70);
    displayText("Hidden Layer");
    auto minmax = Network.minmaxWeightLayer(Reseau::Layer::Hidden);
    for (unsigned i = 0; i < Network.getLayerSize(Reseau::Layer::Hidden); ++i)
    {
        for (unsigned j = 0; j < Network.getLayerSize(Reseau::Layer::Input); ++j)
        {
            float weight = Network.getLayer(Reseau::Layer::Hidden)[i].poids[j];
            float gray_level = (weight - minmax.first) / (minmax.second - minmax.first);
            glColor3f(gray_level, gray_level, gray_level);

            const int side = 3;
            const int posX = 250;
            const int posY = 80;
            auto x = posX + (i % 6) * 70 + (j % 16) * side;
            auto y = posY + (i / 6) * 70 + (j / 16) * side;
            Rectangle(x, y, x + side, y + side, true);
        }
    }

    glColor3f(0, 0, 0);
    glRasterPos2f(750, 70);
    displayText("Output Layer");
    minmax = Network.minmaxWeightLayer(Reseau::Layer::Output);
    for (unsigned i = 0; i < Network.getLayerSize(Reseau::Layer::Output); ++i)
    {
        for (unsigned j = 0; j < Network.getLayerSize(Reseau::Layer::Hidden); ++j)
        {
            float weight = Network.getLayer(Reseau::Layer::Output)[i].poids[j];
            float gray_level = (weight - minmax.first) / (minmax.second - minmax.first);
            glColor3f(gray_level, gray_level, gray_level);

            const int side = 12;
            const int posX = 720;
            const int posY = 100;

            auto x = posX + (i / 3) * 150 + (j % 6) * side;
            auto y = posY + (i % 3) * 150 + (j / 6) * side;
            Rectangle(x, y, x + side, y + side, true);
        }

        glColor3f(0, 0, 0);
        auto typeX = 720 + (i / 3) * 150;
        auto typeY = 90 + (i % 3) * 150;
        glRasterPos2f(typeX, typeY);
        displayText(TypeToString(static_cast<Type>(i)));

        float state = Network.getLayer(Reseau::Layer::Output)[i].etat;
        glColor3f(state, state, state);
        typeY += 90;
        Rectangle(typeX, typeY, typeX + 10, typeY + 10, true);

        float outputState = std::abs(Network.goalOutputData[i]-1);
        glColor3f(outputState, outputState, outputState);
        typeX += 10;
        Rectangle(typeX, typeY, typeX + 10, typeY + 10, true);
    }

    // Mise à jour des statistiques
    auto resultat = Network.get_max_out_index();
    if (currentType == resultat)
    {
        ++GlobalSuccess;
        for (int j = 0; j < 100; ++j)
            ++Success3[j];
    }
    Success3[iteration % 100] = 0;

    glColor3f(0, 0, 0);
    glutSwapBuffers();
}

static void idle_callback()
{
    ++iteration;

    currentType = static_cast<Type>(rand() % 6);

    Network.newInput(currentType);
    Network.feedforward(Reseau::Layer::Hidden);
    Network.feedforward(Reseau::Layer::Output);

    Network.backpropagation();

    glutPostRedisplay();
}

int main(int argc, char** argv)
{
    static int g_Width = 1000;         // Initial window width
    static int g_Height = 500; // Initial window height

    // GLUT Window Initialization:
    glutInit(&argc, argv);
    glutInitWindowSize(g_Width, g_Height);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutCreateWindow("Neural Network GLUT example - LucileM");

    srand( time(0));

    Network.init();
    
    // Register callbacks
    glutDisplayFunc(display);
    glutIdleFunc(idle_callback);

    glOrtho(0, g_Width, g_Height, 0, -1, 1);

    glutMainLoop();

    return 0;
}
