#include <vector>
#include <algorithm>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>

#include "reseau.h"

void Reseau::init()
{
    for (uint8_t layer = 0; layer < 3; ++layer)
    {
        for (auto& node : getLayer(static_cast<Layer>(layer)))
        {
            node.etat = 1.f;

            if (layer != 0)
            {
                node.seuil = (rand() % 100) / 100.0;
                for (int f = 0; f < getLayerSize(static_cast<Layer>(layer - 1)); ++f)
                {
                    node.poids.push_back((rand() % 100) / 1000.0);
                }
            }
        }
    }
}

Type Reseau::get_max_out_index() const
{
    for (unsigned j = 0; j < getLayerSize(Layer::Output) - 1; ++j)
    {
        bool found = true;
        for (int i = j + 1; i < getLayerSize(Layer::Output); ++i)
        {
            if (outputLayer[j].etat <= outputLayer[i].etat)
            {
                found = false;
                break;
            }
        }
        if (found)
            return static_cast<Type>(j);
    }
    return Type::HorizontalLine;
}

void Reseau::newInput(const Type& inputType)
{
    // TODO: Remise à zéro de la couche d'entrée
    goalOutputData.fill(0.f);
    goalOutputData[static_cast<int>(inputType)] = 1.f;

    for (auto& node : inputLayer)
    {
        node.etat = 0.f;
    }

    switch (inputType)
    {
    case Type::HorizontalLine:
    {
        int place = rand() % LARGEUR_IMAGE;
        for (int j = 0; j < LARGEUR_IMAGE; ++j)
            inputLayer[place * LARGEUR_IMAGE + j].etat = 1.0;
        break;
    }
    case Type::VerticalLine:
    {
        int place = rand() % LARGEUR_IMAGE;
        for (int i = 0; i < LARGEUR_IMAGE; ++i)
            inputLayer[i * LARGEUR_IMAGE + place].etat = 1.0;
        break;
    }
    case Type::Square:
    {
        int side = 4;
        int placej = (rand() % (LARGEUR_IMAGE - 10)) + side;
        int placei = (rand() % (LARGEUR_IMAGE - 10)) + side;
        for (int i = -side + placei; i <= side + placei; ++i)
        {
            for (int j = -side + placej; j <= side + placej; ++j)
                inputLayer[i * LARGEUR_IMAGE + j].etat = 1.f;
        }
        break;
    }
    case Type::Triangle:
    {
        int size = rand() % 6;
        int placej = rand() % (LARGEUR_IMAGE - size);
        int placei = rand() % (LARGEUR_IMAGE - size);

        for (int j = 0; j < size; ++j)
        {
            for (int i = 0; i < (size - j); ++i)
            {
                inputLayer[(i + placei) * LARGEUR_IMAGE + j + placej].etat = 1.f;
            }
        }
        break;
    }
    case Type::Cross:
    {
        int size = 2;
        int placej = (rand() % (LARGEUR_IMAGE - 2 * size)) + size;
        int placei = (rand() % (LARGEUR_IMAGE - 2 * size)) + size;
        for (int k = -size; k <= size; ++k)
        {
            inputLayer[placei * LARGEUR_IMAGE + placej + k].etat = 1.f;
            inputLayer[(placei + k)*LARGEUR_IMAGE + placej].etat = 1.f;
        }
        break;
    }
    case Type::CheckedPattern:
    {
        for (int ki = 0; ki < LARGEUR_IMAGE; ++ki)
        {
            for (int kj = 0; kj < LARGEUR_IMAGE; ++kj)
            {
                if (((ki % 2 == 0) && (kj % 2 == 1)) || ((ki % 2 == 1) && (kj % 2 == 0)))
                    inputLayer[ki*LARGEUR_IMAGE + kj].etat = 1.f;
            }
        }
        break;
    }
    }

    for (int i = 0; i < LARGEUR_IMAGE; ++i)
    {
        for (int j = 0; j < LARGEUR_IMAGE; ++j)
        {
            inputLayer[i * LARGEUR_IMAGE + j].etat += (rand() % 100) / 100.f * bruit - bruit / 2.f;
        }
    }
}

std::array<Reseau::Neurone, MAXNODES>& Reseau::getLayer(const Layer& layer)
{
    switch (layer)
    {
        case Layer::Input: return inputLayer;
        case Layer::Hidden: return hiddenLayer;
        case Layer::Output: return outputLayer;
    }
}

const unsigned Reseau::getLayerSize(const Layer& layer) const
{
    switch (layer)
    {
    case Layer::Input: return LARGEUR_IMAGE * LARGEUR_IMAGE;
    case Layer::Hidden: return 36;
    case Layer::Output: return 6;
    }
}

void Reseau::feedforward(const Layer& layer)
{
    for (auto& neuron : getLayer(layer))
    {
        float act = neuron.seuil;

        auto previousLayer = static_cast<Layer>(layer - 1);
        for (unsigned int node = 0; node < getLayerSize(previousLayer); ++node)
        {
            act += getLayer(previousLayer)[node].etat * neuron.poids[node];
        }
        neuron.etat = 1.f / (1.f + exp(-act));
    }
}

tERRORtype Reseau::get_derivees() const
{
    tERRORtype Deriv1;            // dE/dy
    tERRORtype Deriv2;            // dE/ds

    // calcul de dE/dy pour la couche de sortie
    // calculate dE/ds for output nodes 
    for (int node = 0; node < getLayerSize(Layer::Output); ++node)
    {
        auto state = outputLayer[node].etat;
        Deriv1[Layer::Output][node] = goalOutputData[node] - state;
        Deriv2[Layer::Output][node] = Deriv1[Layer::Output][node] * state * (1.0 - state);
    }

    // Calcul de dE/dy et de dE/ds pour la couche intermédiaire.
    for (int hidden_node = 0; hidden_node < getLayerSize(Layer::Hidden); ++hidden_node)
    {
        Deriv1[Layer::Hidden][hidden_node] = 0.f;
        for (int node = 0; node < getLayerSize(Layer::Output); ++node)
        {
            auto weight = outputLayer[node].poids[hidden_node];
            Deriv1[Layer::Hidden][hidden_node] += Deriv2[Layer::Output][node] * weight;
        }
        auto state = hiddenLayer[hidden_node].etat;
        Deriv2[Layer::Hidden][hidden_node] = Deriv1[Layer::Hidden][hidden_node] * state * (1.0 - state);
    }
    return Deriv2;
}

void Reseau::backpropagation()
{
    auto& derivative = get_derivees();

    for (uint8_t layer = 1; layer < 3; ++layer)
    {
        for (int neurone = 0; neurone < getLayerSize(static_cast<Layer>(layer)); ++neurone)
        {
            auto& neuron = getLayer(static_cast<Layer>(layer))[neurone];
            neuron.seuil += EPSILON * derivative[layer][neurone];
            const auto previousLayer = static_cast<Layer>(layer - 1);

            for (int inunit = 0; inunit < getLayerSize(previousLayer); ++inunit)
            {
                auto state = getLayer(previousLayer)[inunit].etat;
                neuron.poids[inunit] += EPSILON * derivative[layer][neurone] * state;
            }
        }
    }
}

std::pair<float, float> Reseau::minmaxWeightLayer(const Layer& layer)
{
    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::min();
    for (auto& neuron : getLayer(layer))
    {
        auto weights = neuron.poids;
        max = std::max(max, *std::max_element(weights.cbegin(), weights.cend()));
        min = std::min(min, *std::min_element(weights.cbegin(), weights.cend()));
    }
    return std::make_pair(min, max);
}
