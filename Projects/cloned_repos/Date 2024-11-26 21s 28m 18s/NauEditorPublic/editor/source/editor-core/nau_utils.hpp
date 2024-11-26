// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Various utility functions

#pragma once

#include "nau/nau_editor_config.hpp"

#include <QJsonObject>
#include <QString>
#include <QVector3D>


namespace Nau::Utils
{
    namespace File
    {
        QJsonObject readJSONFile(const QString& filePath, QString& errorMessage);
    }

    namespace Conversion
    {
        QString timestampToQString(qint64 timestamp);
        QString QSizeToQString(const QSize& size);

        QJsonValue QVector3DToJsonValue(const QVector3D& vector3D);
        QVector3D JsonValueToQVector3D(const QJsonValue& jsonValue, QString& errorMessage);
    }

    namespace Widget
    {
        NAU_EDITOR_API QPoint fitWidgetIntoScreen(const QSize& widgetSize, const QPoint& desiredWidgetPosisition);
    }

    namespace Email
    {
        namespace Data 
        {
            struct EmailCredentials
            {
                QString server;
                QString user;
                QString password;

                QString receivers;

                inline bool isEmpty() const
                {
                    return (server.isEmpty() || user.isEmpty() || password.isEmpty() || receivers.isEmpty());
                }
            };

            EmailCredentials emailData();
        }

        struct EmailRequest
        {
            QString mailFrom;
            QString subject;
            QString body;
            std::function<void(bool)> callback;
        };

        bool sendEmail(const EmailRequest& emailRequest);
    }

}
