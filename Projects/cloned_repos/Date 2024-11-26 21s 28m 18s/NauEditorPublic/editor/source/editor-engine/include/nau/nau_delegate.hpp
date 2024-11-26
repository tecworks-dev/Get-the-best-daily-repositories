// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// All classes for creating delegates inside the editor will be stored here

#pragma once

#include <vector>
#include <functional>
#include <optional>
#undef fatal

class NauCallbackId
{
public:
    NauCallbackId()
        : id(generateId())
    {
    }

    NauCallbackId(const NauCallbackId& other)
        : id(other.id)
    {
    }

    NauCallbackId& operator=(const NauCallbackId& other)
    {
        id = other.id;
        return *this;
    }

    inline bool operator==(const NauCallbackId& other)
    { 
        return id == other.id; 
    }
   
private:
    uint64_t id;

    static uint64_t generateId()
    {
        static uint64_t nextId = 0;       
        return ++nextId;
    }
};


// ** NauDelegate
//
// Base class for creating delegates inside the editor

template <typename ...Args>
class NauDelegate
{
public:
    // Alias for template callback
    using callback = std::function<void(Args...)>;
    using identifiedCallback = std::pair<NauCallbackId, callback>;

    NauDelegate() = default;
    NauDelegate(const class NauDelegate&) = delete;
    NauDelegate& operator=(class NauDelegate&) = delete;

    const NauCallbackId& addCallback(const callback& event)
    {
        m_callbacks.push_back(std::make_pair(NauCallbackId(), event));
        return m_callbacks.back().first;
    }

    void deleteCallback(const NauCallbackId& id)
    {
        for (auto it = m_callbacks.begin(); it != m_callbacks.end(); ++it) {
            if ((*it).first == id) {
                m_callbacks.erase(it);
                break;
            }
        }
    }

    void broadcast(Args... args)
    {
        if (m_broadcastAllowed) {
            for (const identifiedCallback& event : m_callbacks) {
                event.second(args...);
            }
        }
    }

    void disableBroadcast()
    {
        m_broadcastAllowed = false;
    }

    void enableBroadcast()
    {
        m_broadcastAllowed = true;
    }

private:
    std::vector<identifiedCallback> m_callbacks;
    bool m_broadcastAllowed = true;
};
