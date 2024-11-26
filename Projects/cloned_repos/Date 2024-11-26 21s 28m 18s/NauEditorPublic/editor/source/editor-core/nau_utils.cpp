// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_utils.hpp"

#include <QFile>
#include <QSize>

#include <QJsonParseError>
#include <QJsonArray>
#include <QRect>
#include <QGuiApplication>
#include <QScreen>
#include <QPoint>

#include <curl/curl.h>


QJsonObject Nau::Utils::File::readJSONFile(const QString& filePath, QString& errorMessage)
{
    QFile jsonFile(filePath);
    if (!jsonFile.open(QIODevice::ReadOnly)) {
        errorMessage = QString("Failed to read json file {%1}.").arg(filePath);
        return QJsonObject();
    }

    QJsonParseError error;
    const QJsonDocument document = QJsonDocument::fromJson(jsonFile.readAll(), &error);

    jsonFile.close();

    if (error.error != QJsonParseError::NoError) {
        errorMessage = QString("Failed to read json file {%1}.").arg(error.errorString());
        return QJsonObject();
    }

    return document.object();
}

QString Nau::Utils::Conversion::timestampToQString(qint64 timestamp)
{
    QDateTime dateTime = QDateTime::fromSecsSinceEpoch(timestamp);
    return dateTime.toString("dd/MM/yyyy");
}

QString Nau::Utils::Conversion::QSizeToQString(const QSize& size)
{
    return QString("%1x%3").arg(size.width()).arg(size.height());
}

QJsonValue Nau::Utils::Conversion::QVector3DToJsonValue(const QVector3D& vector3D)
{
    QJsonObject json;
    json["x"] = vector3D.x();
    json["y"] = vector3D.y();
    json["z"] = vector3D.z();
    return json;
}

QVector3D Nau::Utils::Conversion::JsonValueToQVector3D(const QJsonValue& value, QString& errorMessage)
{
    if (!value.isObject()) {
        errorMessage = "Expected an object for QVector3D.";
        return QVector3D();
    }

    QJsonObject obj = value.toObject();
    if (!obj.contains("x") || !obj["x"].isDouble() ||
        !obj.contains("y") || !obj["y"].isDouble() ||
        !obj.contains("z") || !obj["z"].isDouble()) {
        errorMessage = "Invalid QVector3D object.";
        return QVector3D();
    }

    return QVector3D(obj["x"].toDouble(), obj["y"].toDouble(), obj["z"].toDouble());
}

QPoint Nau::Utils::Widget::fitWidgetIntoScreen(const QSize& widgetSize, const QPoint& desiredWidgetPosisition)
{
    const QRect screenGeometry = QGuiApplication::primaryScreen()->geometry();

    const int xWidgetPosition = std::max(screenGeometry.left(), std::min(desiredWidgetPosisition.x(), screenGeometry.right() - widgetSize.width()));
    const int yWidgetPosition = std::max(screenGeometry.top(), std::min(desiredWidgetPosisition.y(), screenGeometry.bottom() - widgetSize.height()));

    return QPoint(xWidgetPosition, yWidgetPosition);
}

Nau::Utils::Email::Data::EmailCredentials Nau::Utils::Email::Data::emailData()
{
    // TODO: probably will be removed
    QString errorMessage;
    QJsonObject result = Nau::Utils::File::readJSONFile("TODO", errorMessage);

    if (!errorMessage.isEmpty() || result.isEmpty()) {
        return EmailCredentials();
    }

    EmailCredentials credentials;

    credentials.server = result["server"].toString();
    credentials.user = result["user"].toString();
    credentials.password = result["password"].toString();

    QJsonArray receiversArray = result["receivers"].toArray();
    QStringList receiverStrings;
    for (const QJsonValue& receiver : receiversArray) {
        receiverStrings << receiver.toString();
    }
    credentials.receivers = receiverStrings.join(", ");

    return credentials;
}

bool Nau::Utils::Email::sendEmail(const EmailRequest& emailRequest)
{
    const Nau::Utils::Email::Data::EmailCredentials credentials = Nau::Utils::Email::Data::emailData();

    CURL* curl = curl_easy_init();
    if (!curl || credentials.isEmpty()) {
        if (emailRequest.callback) {
            emailRequest.callback(false);
        }

        return false;
    }

    struct curl_slist* recipients = NULL;

    const std::string smtpServer = credentials.server.toUtf8().constData();
    const std::string username = credentials.user.toUtf8().constData();
    const std::string password = credentials.password.toUtf8().constData();
    const std::string from = emailRequest.mailFrom.toUtf8().constData();
    const std::string to = credentials.receivers.toUtf8().constData();

    const QString dt = QDateTime::currentDateTime().toString(Qt::RFC2822Date);

    const QString body = QString("Date: %1\r\nTo: %2\r\nFrom: %3\r\nSubject: %4\r\n\r\n%5\r\n");
    const std::string text = body.arg(dt).arg(credentials.receivers).arg(emailRequest.mailFrom).arg(emailRequest.subject).arg(emailRequest.body).toUtf8().constData();
    const char* payload_text = text.c_str();

    curl_easy_setopt(curl, CURLOPT_USERNAME, username.c_str());
    curl_easy_setopt(curl, CURLOPT_PASSWORD, password.c_str());
    curl_easy_setopt(curl, CURLOPT_URL, smtpServer.c_str());
    curl_easy_setopt(curl, CURLOPT_MAIL_FROM, from.c_str());

    recipients = curl_slist_append(recipients, to.c_str());
    curl_easy_setopt(curl, CURLOPT_MAIL_RCPT, recipients);

    curl_easy_setopt(curl, CURLOPT_READFUNCTION, +[](void* ptr, size_t size, size_t nmemb, void* userp) -> size_t {
        const char** payload_text = (const char**)userp;
        size_t len = strlen(*payload_text);
        if (len > size * nmemb) {
            len = size * nmemb;
        }
        memcpy(ptr, *payload_text, len);
        *payload_text += len;
        return len;
    });

    curl_easy_setopt(curl, CURLOPT_READDATA, &payload_text);
    curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);

    CURLcode result = curl_easy_perform(curl);
    bool success = (result == CURLE_OK);

    if (emailRequest.callback) {
        emailRequest.callback(success);
    }

    curl_slist_free_all(recipients);
    curl_easy_cleanup(curl);

    return success;
}
