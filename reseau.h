#ifndef RESEAU_H
#define RESEAU_H

#include <vector>
#include <array>

const float EPSILON = 0.50f;
const int LARGEUR_IMAGE = 16;
const int MAXNODES = LARGEUR_IMAGE* LARGEUR_IMAGE;

typedef std::array<std::array<float, MAXNODES>, 3> tERRORtype;

enum class Type
{
    HorizontalLine,
    VerticalLine,
    Square,
    Triangle,
    Cross,
    CheckedPattern,
};

class Reseau
{
public:

    struct Neurone
    {
        std::vector<float> poids; // Poids des connexions d'entrée
        float seuil;              /* threshhold input value        */
        float etat;               /* activation state value of node*/
    };
    enum Layer : uint8_t
    {
        Input = 0,
        Hidden = 1,
        Output = 2,
    };
    tERRORtype get_derivees() const;

    void init();
    void newInput(const Type& inputType);
    void feedforward(const Layer& couche);
    void backpropagation();
    Type get_max_out_index() const;
    std::array<Neurone, MAXNODES>& getLayer(const Layer& layer);
    const unsigned int getLayerSize(const Layer& layer) const ;
    std::pair<float, float> minmaxWeightLayer(const Layer& layer);
    const float getNoise() { return bruit; }
    std::array<float, MAXNODES> goalOutputData;

private:
    std::array<Neurone, MAXNODES> inputLayer;
    std::array<Neurone, MAXNODES> hiddenLayer;
    std::array<Neurone, MAXNODES> outputLayer;
    float bruit = 0.05f;
};

#endif
