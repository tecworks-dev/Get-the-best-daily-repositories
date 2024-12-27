import React from 'react';
import { Clock, Users, Calendar, Tag } from 'lucide-react';

interface PageProps {
  params: {
    id: string;
  };
}

interface Meeting {
  title: string;
  date: string;
  time?: string;
  attendees?: string[];
  tags: string[];
  content: string;
}

export function generateStaticParams() {
  // Return all possible meeting IDs
  return [
    { id: 'team-sync-dec-26' },
    { id: 'product-review' },
    { id: 'project-ideas' },
    { id: 'action-items' }
  ];
}

const MeetingPage = ({ params }: PageProps) => {
  // This would normally come from your database
  const sampleData: Record<string, Meeting> = {
    'team-sync-dec-26': {
      title: 'Team Sync - Dec 26',
      date: '2024-12-26',
      time: '10:00 AM - 11:00 AM',
      attendees: ['John Doe', 'Jane Smith', 'Mike Johnson'],
      tags: ['Team Sync', 'Weekly', 'Product'],
      content: `
# Meeting Summary
Team sync discussion about Q1 2024 goals and current project status.

## Agenda Items
1. Project Status Updates
2. Q1 2024 Planning
3. Team Concerns & Feedback

## Key Decisions
- Prioritized mobile app development for Q1
- Scheduled weekly design reviews
- Added two new features to the roadmap

## Action Items
- [ ] John: Create project timeline
- [ ] Jane: Schedule design review meetings
- [ ] Mike: Update documentation

## Notes
- Discussed current project bottlenecks
- Reviewed customer feedback from last release
- Planned resource allocation for upcoming sprint
      `
    },
    'product-review': {
      title: 'Product Review',
      date: '2024-12-26',
      time: '2:00 PM - 3:00 PM',
      attendees: ['Sarah Wilson', 'Tom Brown', 'Alex Chen'],
      tags: ['Product', 'Review', 'Quarterly'],
      content: `
# Product Review Meeting

## Current Status
- User engagement up by 25%
- New feature adoption rate at 80%
- Customer satisfaction score: 4.5/5

## Key Metrics
1. Monthly Active Users: 50k
2. Average Session Duration: 15 mins
3. Conversion Rate: 12%

## Action Items
- [ ] Sarah: Prepare Q1 roadmap
- [ ] Tom: Analyze user feedback
- [ ] Alex: Update product metrics dashboard

## Next Steps
- Schedule follow-up meetings
- Review competitor analysis
- Plan feature prioritization workshop
      `
    },
    'project-ideas': {
      title: 'Project Ideas',
      date: '2024-12-26',
      time: '4:00 PM - 5:00 PM',
      attendees: ['Emily Lee', 'David Kim', 'Olivia Taylor'],
      tags: ['Project', 'Ideas', 'Brainstorming'],
      content: `
# Project Ideas Meeting

## Current Status
- Reviewed current project pipeline
- Discussed new project ideas

## Key Ideas
1. Develop a new mobile app for customer engagement
2. Create a machine learning model for predictive analytics
3. Design a new website for marketing campaigns

## Action Items
- [ ] Emily: Research mobile app development frameworks
- [ ] David: Investigate machine learning libraries
- [ ] Olivia: Sketch new website design concepts

## Next Steps
- Schedule follow-up meetings to discuss project ideas
- Review project proposals and prioritize projects
- Plan project kick-off meetings
      `
    },
    'action-items': {
      title: 'Action Items Review',
      date: '2024-12-26',
      time: '5:00 PM - 6:00 PM',
      attendees: ['Project Team'],
      tags: ['Tasks', 'Review', 'Planning'],
      content: `
# Action Items Review Meeting

## Progress Review
- Reviewed completed tasks from last week
- Discussed blockers and challenges
- Updated task priorities

## Key Decisions
1. Prioritized security fixes for immediate deployment
2. Scheduled dependency updates for next sprint
3. Assigned new tasks to team members

## Next Steps
- Daily progress updates
- Weekly review meetings
- Monthly planning sessions
      `
    }
  };

  const meeting = sampleData[params.id as keyof typeof sampleData];

  if (!meeting) {
    return <div className="p-8">Meeting not found</div>;
  }

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-4">{meeting.title}</h1>
        
        <div className="flex flex-wrap gap-4 text-gray-600">
          {meeting.date && (
            <div className="flex items-center gap-1">
              <Calendar className="w-4 h-4" />
              <span>{meeting.date}</span>
            </div>
          )}
          
          {meeting.time && (
            <div className="flex items-center gap-1">
              <Clock className="w-4 h-4" />
              <span>{meeting.time}</span>
            </div>
          )}
          
          {meeting.attendees && (
            <div className="flex items-center gap-1">
              <Users className="w-4 h-4" />
              <span>{meeting.attendees.join(', ')}</span>
            </div>
          )}
        </div>

        <div className="flex gap-2 mt-4">
          {meeting.tags.map((tag) => (
            <div key={tag} className="flex items-center gap-1 bg-blue-100 text-blue-800 px-2 py-1 rounded-full text-sm">
              <Tag className="w-3 h-3" />
              {tag}
            </div>
          ))}
        </div>
      </div>

      <div className="prose prose-blue max-w-none">
        <div dangerouslySetInnerHTML={{ __html: meeting.content.split('\n').map(line => {
          if (line.startsWith('# ')) {
            return `<h1>${line.slice(2)}</h1>`;
          } else if (line.startsWith('## ')) {
            return `<h2>${line.slice(3)}</h2>`;
          } else if (line.startsWith('- ')) {
            return `<li>${line.slice(2)}</li>`;
          }
          return line;
        }).join('\n') }} />
      </div>
    </div>
  );
};

export default MeetingPage;
