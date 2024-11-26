// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_input_asset_watcher.hpp"

#include <QJsonArray>
#include <QJsonDocument>
#include <QFile>

#include "nau_log.hpp"
#include "nau/service/service_provider.h"


NauInputAssetWatcher::NauInputAssetWatcher()
{
    m_modifierFactory["Scale"] = [](nau::IInputSystem& insys, const pxr::UsdPrim& prim) {
        return insys.createSignal("scale", "gate", [&](nau::IInputSignal* signal) {
            float value = 0.0f;
            if (prim.GetAttribute(pxr::TfToken("value")).Get(&value)) {
                signal->Properties().set("scale", value);
            }
        });
    };

    m_modifierFactory["DeadZone"] = [](nau::IInputSystem& insys, const pxr::UsdPrim& prim) {
        return insys.createSignal("dead_zone", "gate", [&](nau::IInputSignal* signal) {
            float value = 0.0f;
            if (prim.GetAttribute(pxr::TfToken("value")).Get(&value)) {
                signal->Properties().set("dead_zone", value);
            }
        });
    };

    m_modifierFactory["Clamp"] = [](nau::IInputSystem& insys, const pxr::UsdPrim& prim) {
        return insys.createSignal("clamp", "gate", [&](nau::IInputSignal* signal) {
            float value = 0.0f;
            if (prim.GetAttribute(pxr::TfToken("value")).Get(&value)) {
                signal->Properties().set("clamp", value);
            }
        });
    };

    m_modifierFactory["Multiple"] = [](nau::IInputSystem& insys, const pxr::UsdPrim& prim) {
        return insys.createSignal("multiple", "gate", [&](nau::IInputSignal* signal) {
            float pressingInterval = 0.0;
            if (prim.GetAttribute(pxr::TfToken("pressingInterval")).Get(&pressingInterval)) {
                signal->Properties().set("delay", static_cast<float>(pressingInterval));
            }

            int pressingCount = 0;
            if (prim.GetAttribute(pxr::TfToken("pressingCount")).Get(&pressingCount)) {
                signal->Properties().set("num", pressingCount);
            }
        });
    };

    m_modifierFactory["Delay"] = [](nau::IInputSystem& insys, const pxr::UsdPrim& prim) {
        return insys.createSignal("delay", "gate", [&](nau::IInputSignal* signal) {
            float pressingInterval = 0.0;
            if (prim.GetAttribute(pxr::TfToken("pressingDelay")).Get(&pressingInterval)) {
                signal->Properties().set("delay", static_cast<float>(pressingInterval));
            }
        });
    };

    m_inputSystem = &nau::getServiceProvider().get<nau::IInputSystem>();
}

// ** NauInputAssetWatcher

void NauInputAssetWatcher::addAsset(const std::string& assetPath)
{
    if (isAssetAdded(assetPath)) {
        return;
    }

    m_assetsPath.insert(assetPath);
}

void NauInputAssetWatcher::removeAsset(const std::string& assetPath)
{
    if (!isAssetAdded(assetPath)) {
        return;
    }

    removeAllBind();
    m_stage.Reset();

    m_assetsPath.erase(assetPath);
}

void NauInputAssetWatcher::updateAsset(const std::string& assetPath)
{
    if (!isAssetAdded(assetPath)) {
        return;
    }

    removeAllBind();
    parseAllBinds();
}

void NauInputAssetWatcher::makeAssetCurrent(const std::string& assetPath)
{
    if (!isAssetAdded(assetPath)) {
        return;
    }

    m_stage = PXR_NS::UsdStage::Open(assetPath);
    parseAllBinds();
}

bool NauInputAssetWatcher::isAssetAdded(const std::string& assetPath)
{
    return (m_assetsPath.find(assetPath) != m_assetsPath.end());
}

bool NauInputAssetWatcher::saveToFile(const std::string& path)
{
    QJsonArray jsonArray;

    for (const auto& item : m_assetsPath) {
        jsonArray.append(QString::fromStdString(item));
    }

    QJsonDocument jsonDoc(jsonArray);
    QFile file(QString::fromStdString(path));

    if (!file.open(QIODevice::WriteOnly)) {
        return false;
    }

    file.write(jsonDoc.toJson());
    file.close();

    return true;
}

bool NauInputAssetWatcher::loadFromFile(const std::string& path)
{
    QFile file(QString::fromStdString(path));
    if (!file.open(QIODevice::ReadOnly)) {
        return false;
    }

    QByteArray fileContent = file.readAll();
    file.close();

    QJsonDocument jsonDoc = QJsonDocument::fromJson(fileContent);
    QJsonArray jsonArray = jsonDoc.array();

    for (const auto& jsonValue : jsonArray) {
        const std::string assetPath = jsonValue.toString().toStdString();
        m_assetsPath.insert(assetPath);
        
        m_stage = PXR_NS::UsdStage::Open(assetPath);
        parseAllBinds();
    }

    return true;
}

nau::IInputSignal* NauInputAssetWatcher::createSignal(const pxr::UsdPrim& signalPrim)
{
    nau::IInputSignal* inputSignal = nullptr;

    std::string trigger;
    if (!isAttributeExistAndValid(signalPrim, "trigger", trigger)) {
        return inputSignal;
    }

    std::string device;
    if (!isAttributeExistAndValid(signalPrim, "device", device)) {
        return inputSignal;
    }

    trigger[0] = std::tolower(trigger[0]);
    device[0] = std::tolower(device[0]);

    std::string key;
    if (!isAttributeExistAndValid(signalPrim, "source", key)) {
        return inputSignal;
    }

    bool isMouse = (key == "mouse_x" || key == "mouse_y");
    inputSignal = m_inputSystem->createSignal(isMouse ? "move_relative" : trigger.data(), device.data(), [&](nau::IInputSignal* signal) {
        if (isMouse) {
            signal->Properties().set("axis_x", 0);
            signal->Properties().set("axis_y", 1);
        } else {
            signal->Properties().set("key", eastl::string(key.data()));
        }

        NED_INFO("Added signal: {} by source {}", trigger, key);
    });

    std::string condition;
    if (!isAttributeExistAndValid(signalPrim, "condition", condition)) {
        return inputSignal;
    }

    nau::IInputSignal* modifier = nullptr;
    auto it = m_modifierFactory.find(condition);
    if (it != m_modifierFactory.end()) {
        modifier = it->second(*m_inputSystem, signalPrim);
    }

    if (modifier) {
        modifier->addInput(inputSignal);
        return modifier;
    }

    return inputSignal;
}

nau::IInputSignal* NauInputAssetWatcher::createModifierChain(const pxr::UsdPrim& bindPrim)
{
    nau::IInputSignal* gateSignal = nullptr;
    nau::IInputSignal* lastModifierSignal = nullptr;

    pxr::UsdRelationship modifierArrayRel = bindPrim.GetRelationship(pxr::TfToken("modifierArray"));
    pxr::SdfPathVector modifierPaths;
    modifierArrayRel.GetTargets(&modifierPaths);

    for (const auto& modifierPath : modifierPaths) {
        pxr::UsdPrim modifierPrim = m_stage->GetPrimAtPath(modifierPath);
        
        std::string modifierType;
        if (!isAttributeExistAndValid(modifierPrim, "type", modifierType)) {
            continue;
        }

        nau::IInputSignal* modifierSignal = nullptr;

        auto it = m_modifierFactory.find(modifierType);
        if (it != m_modifierFactory.end()) {
            modifierSignal = it->second(*m_inputSystem, modifierPrim);
        }

        if (modifierSignal) {
            if (!gateSignal) {
                gateSignal = modifierSignal;
            }
            else {
                lastModifierSignal->addInput(modifierSignal);
            }
            lastModifierSignal = modifierSignal;
        }
    }

    return gateSignal;
}

nau::IInputSignal* NauInputAssetWatcher::combineWithModifier(nau::IInputSignal* mainSignal, nau::IInputSignal* modifierSignal)
{
    if (!mainSignal) {
        return nullptr;
    }

    if (modifierSignal) {
        modifierSignal->addInput(mainSignal);
        return modifierSignal;
    }

    return mainSignal;
}

void NauInputAssetWatcher::parseAllBinds()
{
    if (!m_stage || !m_inputSystem) {
        return;
    }

    pxr::UsdPrim bindsPrim = m_stage->GetPrimAtPath(pxr::SdfPath("/Binds"));
    if (!bindsPrim) {
        return;
    }

    for (const auto& bindPrim : bindsPrim.GetChildren()) {
        std::string bindType;
        bindPrim.GetAttribute(pxr::TfToken("type")).Get(&bindType);

        if (bindType == "Digital") {
            parseBind(bindPrim, nau::IInputAction::Type::Trigger);
        } else if (bindType == "Analog") {
            parseBind(bindPrim, nau::IInputAction::Type::Continuous);
        } else if (bindType == "Axis") {
            parseBind(bindPrim, nau::IInputAction::Type::Continuous, true);
        }
    }
}

void NauInputAssetWatcher::removeAllBind()
{
    if (!m_stage || !m_inputSystem) {
        return;
    }

    pxr::UsdPrim bindsPrim = m_stage->GetPrimAtPath(pxr::SdfPath("/Binds"));
    if (!bindsPrim) {
        return;
    }

    for (const auto& bindPrim : bindsPrim.GetChildren()) {
        removeAllForBind(bindPrim);
    }
}

void NauInputAssetWatcher::removeAllForBind(const pxr::UsdPrim& bindPrim)
{
    int bindId;
    if (!bindPrim.GetAttribute(pxr::TfToken("id")).Get(&bindId)) {
        NED_ERROR("Failed to get bind ID from prim.");
        return;
    }

    if (!m_inputSystem->removeAction(std::move(m_actions[bindId]))) {
        NED_ERROR("Failed to remove action for bind ID: {}", bindId);
    }
    m_actions.erase(bindId);
}

void NauInputAssetWatcher::parseBind(const pxr::UsdPrim& bindPrim, nau::IInputAction::Type actionType, bool isAxis /*= false*/)
{
    pxr::SdfPathVector signalPaths;
    int bindId = 0;

    if (!validateAndExtractBindInfo(bindPrim, bindId, signalPaths)) {
        return;
    }

    nau::IInputSignal* primarySignal = nullptr;
    
    if (isAxis) {
        primarySignal = m_inputSystem->createSignal("or", "gate", [](nau::IInputSignal*) {});
        for (const auto& signalPath : signalPaths) {
            auto signal = createSignal(m_stage->GetPrimAtPath(signalPath));
            if (signal) {
                primarySignal->addInput(signal);
            }
        }
    } else {
        primarySignal = createSignal(m_stage->GetPrimAtPath(signalPaths[0]));
    }

    auto modifierSignal = createModifierChain(bindPrim);
    nau::IInputSignal* finalSignal = combineWithModifier(primarySignal, modifierSignal);

    auto action = m_inputSystem->addAction(bindPrim.GetName().data(), actionType, finalSignal, [](nau::IInputSignal* signal) {
    });

    m_actions[bindId] = action;
}

bool NauInputAssetWatcher::validateAndExtractBindInfo(const pxr::UsdPrim& prim, int& bindId, pxr::SdfPathVector& signalPaths)
{
    pxr::UsdRelationship signalArrayRel = prim.GetRelationship(pxr::TfToken("signalArray"));
    signalArrayRel.GetTargets(&signalPaths);

    if (signalPaths.empty()) {
        return false;
    }

    if (!prim.GetAttribute(pxr::TfToken("id")).Get(&bindId)) {
        NED_ERROR("Failed to get bind ID for prim");
        return false;
    }

    if (m_actions.find(bindId) != m_actions.end()) {
        NED_INFO("Action already exists for bind ID: {}", bindId);
        return false;
    }
}

bool NauInputAssetWatcher::isAttributeExistAndValid(const pxr::UsdPrim& prim, const std::string& propertyName, std::string& property)
{
    bool isSuccess = false;

    if (prim.GetAttribute(pxr::TfToken(propertyName)).Get(&property)) {
        isSuccess = true;
    }

    return isSuccess;
}
