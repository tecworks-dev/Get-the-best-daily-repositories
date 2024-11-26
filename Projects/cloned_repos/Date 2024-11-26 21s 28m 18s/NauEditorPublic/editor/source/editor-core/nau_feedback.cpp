// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_feedback.hpp"

#include "themes/nau_theme.hpp"
#include "nau_buttons.hpp"
#include "nau_static_text_label.hpp"
#include "nau_text_edit.hpp"
#include "nau_widget.hpp"
#include "nau_label.hpp"
#include "nau_utils.hpp"


// ** NauThanksForFeedbackDialog

NauThanksForFeedbackDialog::NauThanksForFeedbackDialog(const QString& email,  NauMainWindow* parent) :
    NauDialog(parent)
{
    // Window Title 
    setWindowTitle(tr("Feedback"));

    // Basic Settings
    setFixedSize(WindowSize);

    auto centralLayout = new NauLayoutVertical(this);
    centralLayout->setContentsMargins(Margins);
    centralLayout->setSpacing(VerticalSpacing);

    // Title
    auto titleLabel = new NauLabel(tr("Feedback successfully sent"));
    titleLabel->setFont(Nau::Theme::current().fontFeedbackHeaderThankTitle());
    titleLabel->setWordWrap(true);
    titleLabel->setStyleSheet("color: #FFFFFF");
    centralLayout->addWidget(titleLabel);

    // Content Layout Settings
    auto contentLayout = new NauLayoutHorizontal();
    centralLayout->addLayout(contentLayout);

    // Icon
    auto iconLabel = new NauLabel();
    iconLabel->setPixmap(Nau::Theme::current().iconThanksForFeedbackCheck().pixmap(IconSize));
    contentLayout->addWidget(iconLabel);
    contentLayout->addSpacing(HorizontalSpacing);

    // Content
    QString contentWithoutEmail = QString(tr("Thank you for your feedback, it helps us make our product better."));
    QString contentWithEmail = QString(tr("Thank you for your feedback, it helps us  make our product better. You will get a response via %1")).arg(email);

    auto info = new NauLabel(email.isEmpty() ? contentWithoutEmail : contentWithEmail);
    info->setFont(Nau::Theme::current().fontPrimaryButton());
    info->setWordWrap(true);
    contentLayout->addWidget(info, 0, Qt::AlignVCenter);
    contentLayout->addSpacing(VerticalSpacing);

    auto doneButton = new NauPrimaryButton();
    doneButton->setText(tr("Done"));
    doneButton->setIcon(Nau::Theme::current().iconDonePrimaryStyle());
    doneButton->setFixedSize(DoneButtonSize);
    centralLayout->addWidget(doneButton, 0, Qt::AlignRight);

    connect(doneButton, &NauToolButton::clicked, this, [this]() {
        close();
    });
}

QString NauThanksForFeedbackDialog::myProperty() const
{
    return m_myProperty;
}

void NauThanksForFeedbackDialog::setMyProperty(const QString& value)
{
    if (value != m_myProperty) {
        m_myProperty = value;
        emit eventMyPropertyChanged();
    }
}

// ** NauFeedbackDialog

NauFeedbackDialog::NauFeedbackDialog(NauMainWindow* parent) :
    m_characterCountTemplate(tr("%1 / %2 symbols")),
    m_isErrorExist(true),
    m_isEmailErrorExist(false),
    m_maxCommentLength(1000),
    m_connectionErrorLabel(nullptr),
    m_commentLabel(nullptr),
    m_charCountLabel(nullptr),
    m_errorLabel(nullptr),
    m_commentEdit(nullptr),
    m_emailEdit(nullptr),
    m_sendButton(nullptr),
    NauDialog(parent)
{
    // Window Title 
    setWindowTitle(tr("Feedback"));

    // Basic Settings
    setFixedSize(WindowSize);

    auto centralLayout = new NauLayoutVertical(this);
    centralLayout->setContentsMargins(Margins);

    // Title
    auto titleLabel = new NauStaticTextLabel(tr("Please give us feedback"));
    titleLabel->setFont(Nau::Theme::current().fontFeedbackHeaderTitle());
    titleLabel->setFixedHeight(TitleLabelHeight);
    centralLayout->addWidget(titleLabel);

    m_connectionErrorLabel = new NauStaticTextLabel(tr("You are currently offline, double check connection and try again"));
    m_connectionErrorLabel->setFont(Nau::Theme::current().fontFeedbackConnectionError());
    m_connectionErrorLabel->setColor(ErrorColor);
    m_connectionErrorLabel->setFixedHeight(WidgetHeight);
    m_connectionErrorLabel->setVisible(false);
    centralLayout->addWidget(m_connectionErrorLabel);

    // Comment Info Layout Settings
    auto commentInfoLayout = new NauLayoutHorizontal();
    centralLayout->addLayout(commentInfoLayout);

    m_commentLabel = new NauStaticTextLabel(tr("Your comment"));
    m_commentLabel->setFont(Nau::Theme::current().fontFeedbackCommentInfo());
    m_commentLabel->setColor(SelectedColor);
    m_commentLabel->setFixedHeight(WidgetHeight);
    commentInfoLayout->addWidget(m_commentLabel, 0, Qt::AlignLeft);

    m_charCountLabel = new NauLabel(m_characterCountTemplate.arg("0").arg(QString::number(m_maxCommentLength)));
    m_charCountLabel->setFont(Nau::Theme::current().fontFeedbackErrorInfo());
    m_charCountLabel->setStyleSheet("color: #28D568");
    m_charCountLabel->setFixedHeight(WidgetHeight);
    commentInfoLayout->addWidget(m_charCountLabel, 0, Qt::AlignRight);

    // Comment Section
    m_commentEdit = new NauTextEdit();
    m_commentEdit->setPlaceholderText(tr("Type in your comments here"));
    m_commentEdit->setFixedHeight(CommentEditNormalHeight);
    // Make 1 more symbol to get an error
    m_commentEdit->setMaxLength(m_maxCommentLength + 1);
    centralLayout->addWidget(m_commentEdit);

    // Error Message Section
    m_errorLabel = new NauStaticTextLabel(tr("Please make your text shorter than %1 symbols").arg(QString::number(m_maxCommentLength)));
    m_errorLabel->setFont(Nau::Theme::current().fontFeedbackErrorInfo());
    m_errorLabel->setColor(ErrorColor);
    m_errorLabel->setVisible(false);
    m_errorLabel->setFixedHeight(WidgetHeight);
    centralLayout->addWidget(m_errorLabel);

    // Email Section
    auto emailLayout = new NauLayoutVertical();
    emailLayout->setSpacing(EmailSectionVerticalSpacing);
    centralLayout->addLayout(emailLayout);

    auto emailLabel = new NauStaticTextLabel(tr("E-mail (optional)"));
    emailLabel->setFont(Nau::Theme::current().fontFeedbackCommentInfo());
    emailLabel->setColor(DefaultColor);
    emailLabel->setFixedHeight(WidgetHeight);
    emailLayout->addWidget(emailLabel);

    m_emailEdit = new NauLineEdit();
    m_emailEdit->setPlaceholderText("myname@nauengine.org");
    m_emailEdit->setFixedHeight(EmailEditHeight);
    emailLayout->addWidget(m_emailEdit);

    auto infoLabel = new NauStaticTextLabel(tr("By leaving your email you agree to receive a response from NAU"));
    infoLabel->setFont(Nau::Theme::current().fontFeedbackErrorInfo());
    infoLabel->setColor(DefaultColor);
    infoLabel->setFixedHeight(WidgetHeight);
    emailLayout->addWidget(infoLabel);

    // Footer Section
    auto footerLayout = new NauLayoutHorizontal();
    centralLayout->addLayout(footerLayout);
    
    footerLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Minimum));

    auto cancelButton = new NauSecondaryButton();
    cancelButton->setText(tr("Cancel"));
    cancelButton->setIcon(Nau::Theme::current().iconClosePrimaryStyle());
    cancelButton->setFixedSize(CancelButtonSize);
    footerLayout->addWidget(cancelButton);

    footerLayout->setSpacing(HorizontalSpacingBetweenButton);

    m_sendButton = new NauPrimaryButton();
    m_sendButton->setText(tr("Send"));
    m_sendButton->setIcon(Nau::Theme::current().iconSendPrimaryStyle());
    m_sendButton->setFixedSize(SendButtonSize);
    m_sendButton->setEnabled(false);
    footerLayout->addWidget(m_sendButton);

    connect(m_commentEdit, &NauTextEdit::textChanged, this, &NauFeedbackDialog::onUpdateCharCounter);
    connect(m_commentEdit, &NauTextEdit::eventStateChanged, this, &NauFeedbackDialog::onStateChangedTextEdit);
    connect(m_emailEdit, &NauLineEdit::textChanged, this, &NauFeedbackDialog::onValidateEmail);

    connect(cancelButton, &NauToolButton::clicked, this, [this]() {
        close();
    });
    connect(m_sendButton, &NauPrimaryButton::clicked, this, &NauFeedbackDialog::onSendFeedback);
}

// Request from the auto test team
QString NauFeedbackDialog::myProperty() const
{
    return m_myProperty;
}

// Request from the auto test team
void NauFeedbackDialog::setMyProperty(const QString& value)
{
    if (value != m_myProperty) {
        m_myProperty = value;
        emit eventMyPropertyChanged();
    }
}

QString NauFeedbackDialog::firstNWords(QString text, int nWords)
{
    QStringList words = text.split(' ', Qt::SkipEmptyParts);

    QString result;
    for (int i = 0; i < words.size() && i < nWords; ++i) {
        if (i > 0) {
            result += " ";
        }
        result += words[i];
    }

    return result;
}

void NauFeedbackDialog::updateSendButtonState()
{
    const int length = m_commentEdit->toPlainText().length();

    bool isCommentValid = (length != 0) && (length <= m_maxCommentLength);

    m_sendButton->setEnabled(!m_isEmailErrorExist && isCommentValid);
}

void NauFeedbackDialog::onUpdateCharCounter()
{
    const int length = m_commentEdit->toPlainText().length();
    m_charCountLabel->setText(m_characterCountTemplate.arg(QString::number(length)).arg(QString::number(m_maxCommentLength)));

    m_isErrorExist = ((length != 0) && (length > m_maxCommentLength));
    m_errorLabel->setVisible(m_isErrorExist);

    if (m_isErrorExist) {
        m_commentEdit->setFixedHeight(CommentEditErrorHeight);
        m_commentEdit->setState(NauWidgetState::Error);
    } else {
        m_commentEdit->setFixedHeight(CommentEditNormalHeight);
        m_commentEdit->setState(NauWidgetState::Pressed);
    }

    updateSendButtonState();
}

void NauFeedbackDialog::onValidateEmail()
{
    QRegularExpression rx(R"((^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$))");
    QRegularExpressionValidator validator(rx, this);

    QString email = m_emailEdit->text();
    int pos = 0;

    m_isEmailErrorExist = email.isEmpty() ? false : (validator.validate(email, pos) != QValidator::Acceptable);

    updateSendButtonState();
}

void NauFeedbackDialog::onStateChangedTextEdit(NauWidgetState state)
{
    if (state == NauWidgetState::Active) {
        m_charCountLabel->setStyleSheet("color: #28D568");
        m_commentLabel->setColor(DefaultColor);
    } else if (state == NauWidgetState::Pressed) {
        m_charCountLabel->setStyleSheet("color: #28D568");
        m_commentLabel->setColor(SelectedColor);
    } else if (state == NauWidgetState::Error) {
        m_charCountLabel->setStyleSheet("color: #E66711");
        m_commentLabel->setColor(ErrorColor);
    }
}

void NauFeedbackDialog::onSendFeedback()
{
    const QString email = m_emailEdit->text();
    const QString comment = m_commentEdit->toPlainText();

    Nau::Utils::Email::EmailRequest request;
    request.mailFrom = "info@nauengine.org";
    request.subject = QString("%1 (%2)").arg(QApplication::applicationVersion()).arg(firstNWords(comment, 3));
    request.body = email.isEmpty() ? comment : email + "\n" + comment;

    if (const bool isSuccess = Nau::Utils::Email::sendEmail(request); isSuccess) {
        
        close();
        NauThanksForFeedbackDialog nauThanksForFeedbackDialog(email, nullptr);

        // Request from the auto test team
        [[maybe_unused]] const bool isSuccessThanks
            = nauThanksForFeedbackDialog.setProperty("MyPropertyThanks", QVariant("Hello, World! Thanks"));

        nauThanksForFeedbackDialog.showModal();
    } else {
        m_connectionErrorLabel->setVisible(true);
    }
}
