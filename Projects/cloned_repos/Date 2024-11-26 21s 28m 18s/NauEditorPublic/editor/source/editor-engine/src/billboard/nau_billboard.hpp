//// nau_billboard.hpp
////
//// Copyright © 2023 N-GINN LLC. All rights reserved.
////
//// Billboard utils
//
//#pragma once
//
//#include "scene/nau_editor_engine_scene.hpp"
//
//#include <math/dag_TMatrix.h>
//
//
//// ** NauLightBillboardRenderer
////
//// Provides billboard renderer functions for light
//
//class NauLightBillboardRenderer
//{
//public:
//    NauLightBillboardRenderer() = delete;
//    static void renderBillboards();
//
//private:
//    static void renderBillboard(const TMatrix& position);
//};
//
//
//// ** NauBillboardContainer
////
//// Container with objects that refers to objects from editor scene
//
//using NauBillboardList = std::vector<NauSceneObjectPtr>;
//
//class NauBillboardContainer
//{
//public:
//    NauBillboardContainer();
//    const NauBillboardList& billboards() const { return m_billboardsObjects; }
//
//    void addBillboard(NauSceneObjectPtr object);
//    void removeBillboard(NauEngineObjectID object);
//    void clear();
//
//private:
//    NauBillboardList m_billboardsObjects;
//};
