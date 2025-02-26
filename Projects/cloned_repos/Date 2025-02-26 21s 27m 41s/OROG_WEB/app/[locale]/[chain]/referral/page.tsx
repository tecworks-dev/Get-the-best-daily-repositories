const gradeArr = [
    {
        label: 'LV.1',
        value: '30%',
        h: '9rem'
    },
    {
        label: 'LV.2',
        value: '40%',
        h: '11.1875rem'
    }, {
        label: 'LV.3',
        value: '50%',
        h: '14.1875rem'
    }, {
        label: 'LV.4',
        value: '60%',
        h: '17.25rem'
    }
]
const Referral = () => {
    return (
        <div className="pt-16 flex items-start justify-between px-13.75">
            <div className="text-34.45 text-white">
                <div className="">
                    <p>邀请好友，</p>
                    <p>最多可赚取高达<span className="bg-6FFF89-22FFFF    bg-clip-text text-transparent text-5xl font-bold font">75%</span>的返佣！</p>
                </div>
            </div>
        </div>
    )
}

export default Referral