// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract VotingDApp {
    struct Candidate {
        string name;
        uint voteCount;
    }

    address public owner;
    mapping(address => bool) public hasVoted;
    Candidate[] public candidates;

    event Voted(address indexed voter, uint candidateIndex);

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this");
        _;
    }

    constructor(string[] memory candidateNames) {
        owner = msg.sender;
        for (uint i = 0; i < candidateNames.length; i++) {
            candidates.push(Candidate(candidateNames[i], 0));
        }
    }

    function vote(uint candidateIndex) external {
        require(!hasVoted[msg.sender], "You have already voted");
        require(candidateIndex < candidates.length, "Invalid candidate");
        
        hasVoted[msg.sender] = true;
        candidates[candidateIndex].voteCount++;

        emit Voted(msg.sender, candidateIndex);
    }

    function getCandidates() external view returns (Candidate[] memory) {
        return candidates;
    }
}
