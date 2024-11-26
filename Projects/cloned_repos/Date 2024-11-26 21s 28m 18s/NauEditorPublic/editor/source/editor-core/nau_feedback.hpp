// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Feedback dialog for processing requests from users

#pragma once

#include "nau_widget.hpp"


class NauTextEdit;
class NauLabel;
class NauStaticTextLabel;
class NauPrimaryButton;


// ** NauThanksForFeedbackDialog

class NauThanksForFeedbackDialog : public NauDialog
{
    Q_OBJECT
    // Request from the auto test team
    Q_PROPERTY(QString myProperty READ myProperty WRITE setMyProperty NOTIFY eventMyPropertyChanged)

public:
    NauThanksForFeedbackDialog(const QString& email, NauMainWindow* parent);

    // Request from the auto test team
    QString myProperty() const;
    void setMyProperty(const QString& value);

private:
    // Request from the auto test team
    QString m_myProperty;

private:
    inline static constexpr QSize WindowSize = QSize(455, 248);

    inline static constexpr QMargins Margins = QMargins(32, 32, 32, 32);
    inline static constexpr int VerticalSpacing = 16;
    inline static constexpr int HorizontalSpacing = 23;

    inline static constexpr QSize IconSize = QSize(80, 80);
    inline static constexpr QSize DoneButtonSize = QSize(89, 33);

    // Request from the auto test team
signals:
    void eventMyPropertyChanged();
};

// ** NauFeedbackDialog

class NauFeedbackDialog : public NauDialog
{
    Q_OBJECT
    // Request from the auto test team
    Q_PROPERTY(QString myProperty READ myProperty WRITE setMyProperty NOTIFY eventMyPropertyChanged)

public:
    NauFeedbackDialog(NauMainWindow* parent);

    // Request from the auto test team
    QString myProperty() const;
    void setMyProperty(const QString& value);

private:
    QString firstNWords(QString text, int nWords);
    void updateSendButtonState();

private:
    bool m_isErrorExist;
    bool m_isEmailErrorExist;
    int m_maxCommentLength;

    QString m_characterCountTemplate;

    // Request from the auto test team
    QString m_myProperty;

private:
    inline static constexpr NauColor DefaultCharCountColor = NauColor(43, 210, 108);
    inline static constexpr NauColor SelectedColor = NauColor(85, 103, 255);
    inline static constexpr NauColor ErrorColor = NauColor(227, 105, 21);
    inline static constexpr NauColor DefaultColor = NauColor(128, 128, 128);

    inline static constexpr QSize WindowSize = QSize(455, 544);

    inline static constexpr QMargins Margins = QMargins(16, 16, 16, 16);

    inline static constexpr int TitleLabelHeight = 20;
    inline static constexpr int WidgetHeight = 16;

    inline static constexpr int CommentEditNormalHeight = 300;
    inline static constexpr int CommentEditErrorHeight = 276;

    inline static constexpr int EmailSectionVerticalSpacing = 8;

    inline static constexpr int EmailEditHeight = 32;

    inline static constexpr QSize CancelButtonSize = QSize(99, 32);
    inline static constexpr int HorizontalSpacingBetweenButton = 8;
    inline static constexpr QSize SendButtonSize = QSize(134, 32);

private:
    NauStaticTextLabel* m_connectionErrorLabel;

    NauStaticTextLabel* m_commentLabel;
    NauLabel* m_charCountLabel;

    NauTextEdit* m_commentEdit;

    NauStaticTextLabel* m_errorLabel;

    NauLineEdit* m_emailEdit;

    NauPrimaryButton* m_sendButton;

    // Request from the auto test team
signals:
    void eventMyPropertyChanged();

private slots:
    void onUpdateCharCounter();
    void onValidateEmail();
    void onStateChangedTextEdit(NauWidgetState state);
    void onSendFeedback();
};
