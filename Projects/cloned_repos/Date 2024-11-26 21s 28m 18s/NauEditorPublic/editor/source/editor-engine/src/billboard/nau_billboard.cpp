//#include "nau_billboard.hpp"
//#include <debug/dag_debug3d.h>
//#include <camera/sceneCam.h>
//#include "gameEditor/nau_game_editor.hpp"
//#include "math/dag_mathUtils.h"
//
//
//// ** NauLightBillboardRenderer
////
//// Render Billboard for Light Sources.
//
//void NauLightBillboardRenderer::renderBillboard(const TMatrix& position)
//{
//    constexpr float wireframeScale = 0.025f;
//    const E3DCOLOR wireframeColor = E3DCOLOR(255, 255, 255, 255);
//
//    static const std::vector<Point3> lightBulbPoints = {
//        Point3(-5.5, 3, 0) * wireframeScale,
//        Point3(-2.5, 6, 0) * wireframeScale,
//        Point3(2.5, 6, 0) * wireframeScale,
//        Point3(5.5, 3, 0) * wireframeScale,
//        Point3(5.5, -1, 0) * wireframeScale,
//        Point3(3.5, -3, 0) * wireframeScale,
//        Point3(3.5, -5, 0) * wireframeScale,
//        Point3(2.5, -6, 0) * wireframeScale,
//        Point3(2.5, -8, 0) * wireframeScale,
//        Point3(1.5, -9, 0) * wireframeScale,
//        Point3(-1.5, -9, 0) * wireframeScale,
//        Point3(-2.5, -8, 0) * wireframeScale,
//        Point3(-2.5, -6, 0) * wireframeScale,
//        Point3(-3.5, -5, 0) * wireframeScale,
//        Point3(-3.5, -3, 0) * wireframeScale,
//        Point3(-5.5, -1, 0) * wireframeScale
//    };
//
//    static const std::vector<std::pair<Point3, Point3>> rayLines = {
//        { Point3(6.5, 1, 0) * wireframeScale, Point3(9.5, 1, 0) * wireframeScale },
//        { Point3(4.5, 5, 0) * wireframeScale, Point3(6.5, 7, 0) * wireframeScale },
//        { Point3(0, 7, 0) * wireframeScale, Point3(0, 10, 0) * wireframeScale },
//        { Point3(-4.5, 5, 0) * wireframeScale, Point3(-6.5, 7, 0) * wireframeScale },
//        { Point3(-6.5, 1, 0) * wireframeScale, Point3(-9.5, 1, 0) * wireframeScale }
//    };
//    
//    TMatrix lookAtMat;
//    lookAt(position.col[3], ::get_current_camera_itm().col[3], Point3(0, 1, 0), lookAtMat);
//    lookAtMat.col[1] = Point3(0, 1, 0);
//
//    for (int i = 0; i < lightBulbPoints.size() - 1; ++i) {
//        draw_cached_debug_line(lightBulbPoints[i] * lookAtMat, lightBulbPoints[i + 1] * lookAtMat, wireframeColor);
//    }
//
//    draw_cached_debug_line(lightBulbPoints[lightBulbPoints.size() - 1] * lookAtMat, lightBulbPoints[0] * lookAtMat, wireframeColor);
//    draw_cached_debug_line(lightBulbPoints[7] * lookAtMat, lightBulbPoints[12] * lookAtMat, wireframeColor);
//
//    for (const auto& ray : rayLines) {
//        draw_cached_debug_line(ray.first * lookAtMat, ray.second * lookAtMat, wireframeColor);
//    }
//}
//
//void NauLightBillboardRenderer::renderBillboards()
//{
//    NauBillboardContainer& container = NauGameEditor::editor().billboardContainer();
//    if (container.billboards().empty()) {
//        return;
//    }
//
//    ::begin_draw_cached_debug_lines(true, true);
//    for (const auto object : container.billboards()) {
//        const TMatrix transform = object->component<TMatrix>("transform");
//        renderBillboard(transform);
//    }
//    ::end_draw_cached_debug_lines();
//}
//
//
//// ** NauBillboardContainer
//
//NauBillboardContainer::NauBillboardContainer()
//{
//    NauEngineSceneDelegates::onObjectCreated.addCallback([this](NauSceneObjectPtr sceneObject) {
//        // Now only these two templates have billboards
//        // TODO: Refactor in a new version of the engine when the templates are removed
//        if ((sceneObject->meta().typeName == "nSpotLight") || (sceneObject->meta().typeName == "nOmniLight")) {
//            addBillboard(sceneObject);
//        }
//    });
//
//    NauEngineSceneDelegates::onObjectRemoved.addCallback([this](NauEngineObjectID id) {
//        removeBillboard(id);
//    });
//
//}
//
//void NauBillboardContainer::addBillboard(NauSceneObjectPtr object)
//{
//    m_billboardsObjects.push_back(object);
//}
//
//void NauBillboardContainer::removeBillboard(NauEngineObjectID id)
//{
//    std::erase_if(m_billboardsObjects, [&id](NauSceneObjectPtr selected) {
//        return selected->id() == id;
//    });
//}
//
//void NauBillboardContainer::clear()
//{
//    m_billboardsObjects.clear();
//}