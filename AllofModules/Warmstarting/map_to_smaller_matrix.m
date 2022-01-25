function [G_new ] = map_to_smaller_matrix(G, eind_pre, eind_new)
global cnstData
    pre_nSDP                   = cnstData.pre_nSDP;
    nSDP                       = cnstData.nSDP;
    n_S                        = cnstData.n_S;
    G_new                      = zeros(nSDP);
    G_new(1:n_S,1:n_S) = G(1:n_S,1:n_S);
    G_new( eind_new,   1:n_S ) = G(eind_pre, 1:n_S); 
    G_new( 1:n_S   ,eind_new ) = G(1:n_S   , eind_pre); 
    G_new( eind_new, eind_new) = G(eind_pre,eind_pre);

    G_new( eind_new, nSDP    ) = G(eind_pre, pre_nSDP);
    G_new( 1:n_S   , nSDP    ) = G(1:n_S   , pre_nSDP);
    G_new(     nSDP, eind_new) = G(pre_nSDP, eind_pre);
    G_new(     nSDP, 1:n_S   ) = G(pre_nSDP, 1:n_S   );
%         G_new( 1:n_S, eind2) = rowappend';%         G_new( eind2, eind2) = squareappend;
%         G_new( eind2, nSDP ) = G(eind2, pre_nSDP);
%         G_new( nSDP , eind2) = G(pre_nSDP, eind2);        
end