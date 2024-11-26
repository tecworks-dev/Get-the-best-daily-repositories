// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Create, serialize and deserialize of vfx data


#pragma once

#include <string>

class NauVFX;

// ** NauVFXFile

class NauVFXFile final
{
public:
    NauVFXFile() = delete;
    explicit NauVFXFile(std::string filePath) noexcept;

    void createVFX() const;
    bool loadVFX(NauVFX& vfx) const;
    void saveVFX(const NauVFX& vfx) const;

private:
    std::string m_filePath;
};
