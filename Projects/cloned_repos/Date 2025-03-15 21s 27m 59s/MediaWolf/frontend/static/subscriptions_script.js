document.addEventListener('DOMContentLoaded', () => {
    initializeSubscriptionsPage();
});

function initializeSubscriptionsPage() {
    socket.emit("request_subs")
    const subscriptionList = document.getElementById('subscription-list');
    const rowTemplate = document.getElementById('subscription-row-template');

    function addSubscriptionRow(subscription) {
        const row = rowTemplate.content.cloneNode(true);
        row.querySelector('.subscription-name').textContent = subscription.name;
        row.querySelector('.subscription-last-synced').textContent = subscription.lastSync;
        row.querySelector('.subscription-item-count').textContent = subscription.items;

        row.querySelector('.edit-button').addEventListener('click', () => {
            alert(`Edit: ${subscription.name}`);
        });

        row.querySelector('.remove-button').addEventListener('click', () => {
            alert(`Remove: ${subscription.name}`);
        });

        subscriptionList.appendChild(row);
    }

    socket.on("subs_list", (data) => {
        subscriptionList.innerHTML = "";
        data.subscriptions.forEach(addSubscriptionRow);
    });
}