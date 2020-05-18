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
    void updateStats();
    std::array<Neurone, MAXNODES>& getLayer(const Layer& layer);
    const unsigned int getLayerSize(const Layer& layer) const ;
    std::pair<float, float> minmaxWeightLayer(const Layer& layer);
    const float getNoise() { return _bruit; }
    std::array<float, MAXNODES> goalOutputData;
    Type& getType() { return _type; }
    int getInstantSuccess() { return _success[_iteration % 100]; }
    int getGlobalSuccess() { return _globalSuccess * 100.0 / _iteration; }
    int getIteration() { return _iteration; }
private:
    std::array<Neurone, MAXNODES> _inputLayer;
    std::array<Neurone, MAXNODES> _hiddenLayer;
    std::array<Neurone, MAXNODES> _outputLayer;
    float _bruit = 0.05f;
    Type _type;
    int _globalSuccess;
    std::array<int, 100> _success;
    int _iteration;
};

#endif
