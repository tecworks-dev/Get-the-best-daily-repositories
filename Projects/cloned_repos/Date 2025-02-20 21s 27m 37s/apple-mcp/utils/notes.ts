import { run } from '@jxa/run';

async function getAllNotes() {
    const notes: { [key: string]: string } = await run(() => {
        const Notes = Application('Notes');
        const notes = Notes.notes();

        return notes.map((note: any) => ({
            name: note.name(),
            content: note.plaintext()
        }));
    });

    return notes;
}


async function findNote(searchText: string) {
    const notes: { name: string, content: string }[] = await run((searchText: string) => {
        const Notes = Application('Notes');
        const notes = Notes.notes.whose({_or: [
            {name: {_contains: searchText}},
            {plaintext: {_contains: searchText}}
        ]})()
        return notes.length > 0 ? notes.map((note: any) => ({
            name: note.name(),
            content: note.plaintext()
        })) : [];
    }, searchText);

    if (notes.length === 0) {
        const allNotes = await getAllNotes();
        const closestMatch = Object.entries(allNotes).find(([name]) => 
            name.toLowerCase().includes(searchText.toLowerCase())
        );
        return closestMatch ? [{
            name: closestMatch[0],
            content: closestMatch[1]
        }] : [];
    }

    return notes;
}

export default { getAllNotes, findNote };

