
<template>
    <modal v-slot="{ close }">
        <ReferenceNameForm
            :initial_name="reference.name"
            label="Rename branch"
            @submit="submit($event).then(close)"
        />
    </modal>
</template>

<script>
    import ReferenceNameForm from '@/forms/ReferenceNameForm';

    export default {
        components: { ReferenceNameForm },
        inject: [
            'repo', 'references', 'hidden_references',
            'setSelectedReference', 'refreshHistory',
        ],
        props: {
            reference: { type: Object, required: true },
        },
        methods: {
            async submit(data) {
                const existing_branch = _.find(this.references, { type: 'local_branch', name: data.name });
                const msg = existing_branch && `Overwritten local branch: ${data.name} (was ${existing_branch.hash})`;
                await this.repo.callGit('branch', '--move', this.reference.name, data.name, ...data.force ? ['--force'] : [], { msg });
                await this.refreshHistory();

                const new_reference = _.find(this.references, { type: this.reference.type, name: data.name });
                this.setSelectedReference(new_reference);

                if (this.hidden_references.has(this.reference.id)) {
                    this.hidden_references.delete(this.reference.id);
                    this.hidden_references.add(new_reference.id);
                }
            },
        },
    };
</script>
