import { run } from '@jxa/run';

async function getAllNumbers() {
    const nums: { [key: string]: string[] } = await run(() => {
        const Contacts = Application('Contacts');
        const people = Contacts.people();
        const phoneNumbers: { [key: string]: string[] } = {};

        for (const person of people) {
            const name = person.name();
            const phones = person.phones().map((phone: any) => phone.value());

            if (!phoneNumbers[name]) {
                phoneNumbers[name] = [];
            }
            phoneNumbers[name] = [...phoneNumbers[name], ...phones];
        }

        return phoneNumbers;
    });

    return nums;
}

async function findNumber(name: string) {
    const nums: string[] = await run((name: string) => {
        const Contacts = Application('Contacts');
        const people = Contacts.people.whose({name: {_contains: name}});
        const phones = people.length > 0 ? people[0].phones() : [];
        return phones.map((phone: any) => phone.value());
    }, name);

    // If no numbers found, run getNumbers() to find the closest match
    if (nums.length === 0) {
        const allNumbers = await getAllNumbers();
        const closestMatch = Object.keys(allNumbers).find(personName => 
            personName.toLowerCase().includes(name.toLowerCase())
        );
        return closestMatch ? allNumbers[closestMatch] : [];
    }

    return nums;
}

export default { getAllNumbers, findNumber };
